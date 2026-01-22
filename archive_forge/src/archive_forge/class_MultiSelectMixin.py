from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Generic, Sequence, cast, overload
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.MultiSelect_pb2 import MultiSelect as MultiSelectProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id, save_for_app_testing
from streamlit.type_util import (
class MultiSelectMixin:

    @gather_metrics('multiselect')
    def multiselect(self, label: str, options: OptionSequence[T], default: Any | None=None, format_func: Callable[[Any], Any]=str, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, max_selections: int | None=None, placeholder: str='Choose an option', disabled: bool=False, label_visibility: LabelVisibility='visible') -> list[T]:
        """Display a multiselect widget.
        The multiselect widget starts as empty.

        Parameters
        ----------
        label : str
            A short label explaining to the user what this select widget is for.
            The label can optionally contain Markdown and supports the following
            elements: Bold, Italics, Strikethroughs, Inline Code, Emojis, and Links.

            This also supports:

            * Emoji shortcodes, such as ``:+1:``  and ``:sunglasses:``.
              For a list of all supported codes,
              see https://share.streamlit.io/streamlit/emoji-shortcodes.

            * LaTeX expressions, by wrapping them in "$" or "$$" (the "$$"
              must be on their own lines). Supported LaTeX functions are listed
              at https://katex.org/docs/supported.html.

            * Colored text, using the syntax ``:color[text to be colored]``,
              where ``color`` needs to be replaced with any of the following
              supported colors: blue, green, orange, red, violet, gray/grey, rainbow.

            Unsupported elements are unwrapped so only their children (text contents) render.
            Display unsupported elements as literal characters by
            backslash-escaping them. E.g. ``1\\. Not an ordered list``.

            For accessibility reasons, you should never set an empty label (label="")
            but hide it with label_visibility if needed. In the future, we may disallow
            empty labels by raising an exception.
        options : Iterable
            Labels for the select options in an Iterable. For example, this can
            be a list, numpy.ndarray, pandas.Series, pandas.DataFrame, or
            pandas.Index. For pandas.DataFrame, the first column is used.
            Each label will be cast to str internally by default.
        default: Iterable of V, V, or None
            List of default values. Can also be a single value.
        format_func : function
            Function to modify the display of selectbox options. It receives
            the raw option as an argument and should output the label to be
            shown for that option. This has no impact on the return value of
            the multiselect.
        key : str or int
            An optional string or integer to use as the unique key for the widget.
            If this is omitted, a key will be generated for the widget
            based on its content. Multiple widgets of the same type may
            not share the same key.
        help : str
            An optional tooltip that gets displayed next to the multiselect.
        on_change : callable
            An optional callback invoked when this multiselect's value changes.
        args : tuple
            An optional tuple of args to pass to the callback.
        kwargs : dict
            An optional dict of kwargs to pass to the callback.
        max_selections : int
            The max selections that can be selected at a time.
        placeholder : str
            A string to display when no options are selected. Defaults to 'Choose an option'.
        disabled : bool
            An optional boolean, which disables the multiselect widget if set
            to True. The default is False. This argument can only be supplied
            by keyword.
        label_visibility : "visible", "hidden", or "collapsed"
            The visibility of the label. If "hidden", the label doesn't show but there
            is still empty space for it above the widget (equivalent to label="").
            If "collapsed", both the label and the space are removed. Default is
            "visible".

        Returns
        -------
        list
            A list with the selected options

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> options = st.multiselect(
        ...     'What are your favorite colors',
        ...     ['Green', 'Yellow', 'Red', 'Blue'],
        ...     ['Yellow', 'Red'])
        >>>
        >>> st.write('You selected:', options)

        .. output::
           https://doc-multiselect.streamlit.app/
           height: 420px

        """
        ctx = get_script_run_ctx()
        return self._multiselect(label=label, options=options, default=default, format_func=format_func, key=key, help=help, on_change=on_change, args=args, kwargs=kwargs, max_selections=max_selections, placeholder=placeholder, disabled=disabled, label_visibility=label_visibility, ctx=ctx)

    def _multiselect(self, label: str, options: OptionSequence[T], default: Sequence[Any] | Any | None=None, format_func: Callable[[Any], Any]=str, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, max_selections: int | None=None, placeholder: str='Choose an option', disabled: bool=False, label_visibility: LabelVisibility='visible', ctx: ScriptRunContext | None=None) -> list[T]:
        key = to_key(key)
        check_callback_rules(self.dg, on_change)
        check_session_state_rules(default_value=default, key=key)
        opt = ensure_indexable(options)
        check_python_comparable(opt)
        maybe_raise_label_warnings(label, label_visibility)
        indices = _check_and_convert_to_indices(opt, default)
        id = compute_widget_id('multiselect', user_key=key, label=label, options=[str(format_func(option)) for option in opt], default=indices, key=key, help=help, max_selections=max_selections, placeholder=placeholder, form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
        default_value: list[int] = [] if indices is None else indices
        multiselect_proto = MultiSelectProto()
        multiselect_proto.id = id
        multiselect_proto.label = label
        multiselect_proto.default[:] = default_value
        multiselect_proto.options[:] = [str(format_func(option)) for option in opt]
        multiselect_proto.form_id = current_form_id(self.dg)
        multiselect_proto.max_selections = max_selections or 0
        multiselect_proto.placeholder = placeholder
        multiselect_proto.disabled = disabled
        multiselect_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
        if help is not None:
            multiselect_proto.help = dedent(help)
        serde = MultiSelectSerde(opt, default_value)
        widget_state = register_widget('multiselect', multiselect_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
        default_count = _get_default_count(widget_state.value)
        if max_selections and default_count > max_selections:
            raise StreamlitAPIException(_get_over_max_options_message(default_count, max_selections))
        widget_state = maybe_coerce_enum_sequence(widget_state, options, opt)
        if widget_state.value_changed:
            multiselect_proto.value[:] = serde.serialize(widget_state.value)
            multiselect_proto.set_value = True
        if ctx:
            save_for_app_testing(ctx, id, format_func)
        self.dg._enqueue('multiselect', multiselect_proto)
        return widget_state.value

    @property
    def dg(self) -> DeltaGenerator:
        """Get our DeltaGenerator."""
        return cast('DeltaGenerator', self)