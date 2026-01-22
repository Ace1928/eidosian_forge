from __future__ import annotations
from asyncio import get_running_loop
from contextlib import contextmanager
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Callable, Generic, Iterator, TypeVar, Union, cast
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.auto_suggest import AutoSuggest, DynamicAutoSuggest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.clipboard import Clipboard, DynamicClipboard, InMemoryClipboard
from prompt_toolkit.completion import Completer, DynamicCompleter, ThreadedCompleter
from prompt_toolkit.cursor_shapes import (
from prompt_toolkit.document import Document
from prompt_toolkit.enums import DEFAULT_BUFFER, SEARCH_BUFFER, EditingMode
from prompt_toolkit.eventloop import InputHook
from prompt_toolkit.filters import (
from prompt_toolkit.formatted_text import (
from prompt_toolkit.history import History, InMemoryHistory
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding.bindings.auto_suggest import load_auto_suggest_bindings
from prompt_toolkit.key_binding.bindings.completion import (
from prompt_toolkit.key_binding.bindings.open_in_editor import (
from prompt_toolkit.key_binding.key_bindings import (
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Float, FloatContainer, HSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer, WindowAlign
from prompt_toolkit.layout.controls import (
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.menus import CompletionsMenu, MultiColumnCompletionsMenu
from prompt_toolkit.layout.processors import (
from prompt_toolkit.layout.utils import explode_text_fragments
from prompt_toolkit.lexers import DynamicLexer, Lexer
from prompt_toolkit.output import ColorDepth, DummyOutput, Output
from prompt_toolkit.styles import (
from prompt_toolkit.utils import (
from prompt_toolkit.validation import DynamicValidator, Validator
from prompt_toolkit.widgets.toolbars import (
class PromptSession(Generic[_T]):
    """
    PromptSession for a prompt application, which can be used as a GNU Readline
    replacement.

    This is a wrapper around a lot of ``prompt_toolkit`` functionality and can
    be a replacement for `raw_input`.

    All parameters that expect "formatted text" can take either just plain text
    (a unicode object), a list of ``(style_str, text)`` tuples or an HTML object.

    Example usage::

        s = PromptSession(message='>')
        text = s.prompt()

    :param message: Plain text or formatted text to be shown before the prompt.
        This can also be a callable that returns formatted text.
    :param multiline: `bool` or :class:`~prompt_toolkit.filters.Filter`.
        When True, prefer a layout that is more adapted for multiline input.
        Text after newlines is automatically indented, and search/arg input is
        shown below the input, instead of replacing the prompt.
    :param wrap_lines: `bool` or :class:`~prompt_toolkit.filters.Filter`.
        When True (the default), automatically wrap long lines instead of
        scrolling horizontally.
    :param is_password: Show asterisks instead of the actual typed characters.
    :param editing_mode: ``EditingMode.VI`` or ``EditingMode.EMACS``.
    :param vi_mode: `bool`, if True, Identical to ``editing_mode=EditingMode.VI``.
    :param complete_while_typing: `bool` or
        :class:`~prompt_toolkit.filters.Filter`. Enable autocompletion while
        typing.
    :param validate_while_typing: `bool` or
        :class:`~prompt_toolkit.filters.Filter`. Enable input validation while
        typing.
    :param enable_history_search: `bool` or
        :class:`~prompt_toolkit.filters.Filter`. Enable up-arrow parting
        string matching.
    :param search_ignore_case:
        :class:`~prompt_toolkit.filters.Filter`. Search case insensitive.
    :param lexer: :class:`~prompt_toolkit.lexers.Lexer` to be used for the
        syntax highlighting.
    :param validator: :class:`~prompt_toolkit.validation.Validator` instance
        for input validation.
    :param completer: :class:`~prompt_toolkit.completion.Completer` instance
        for input completion.
    :param complete_in_thread: `bool` or
        :class:`~prompt_toolkit.filters.Filter`. Run the completer code in a
        background thread in order to avoid blocking the user interface.
        For ``CompleteStyle.READLINE_LIKE``, this setting has no effect. There
        we always run the completions in the main thread.
    :param reserve_space_for_menu: Space to be reserved for displaying the menu.
        (0 means that no space needs to be reserved.)
    :param auto_suggest: :class:`~prompt_toolkit.auto_suggest.AutoSuggest`
        instance for input suggestions.
    :param style: :class:`.Style` instance for the color scheme.
    :param include_default_pygments_style: `bool` or
        :class:`~prompt_toolkit.filters.Filter`. Tell whether the default
        styling for Pygments lexers has to be included. By default, this is
        true, but it is recommended to be disabled if another Pygments style is
        passed as the `style` argument, otherwise, two Pygments styles will be
        merged.
    :param style_transformation:
        :class:`~prompt_toolkit.style.StyleTransformation` instance.
    :param swap_light_and_dark_colors: `bool` or
        :class:`~prompt_toolkit.filters.Filter`. When enabled, apply
        :class:`~prompt_toolkit.style.SwapLightAndDarkStyleTransformation`.
        This is useful for switching between dark and light terminal
        backgrounds.
    :param enable_system_prompt: `bool` or
        :class:`~prompt_toolkit.filters.Filter`. Pressing Meta+'!' will show
        a system prompt.
    :param enable_suspend: `bool` or :class:`~prompt_toolkit.filters.Filter`.
        Enable Control-Z style suspension.
    :param enable_open_in_editor: `bool` or
        :class:`~prompt_toolkit.filters.Filter`. Pressing 'v' in Vi mode or
        C-X C-E in emacs mode will open an external editor.
    :param history: :class:`~prompt_toolkit.history.History` instance.
    :param clipboard: :class:`~prompt_toolkit.clipboard.Clipboard` instance.
        (e.g. :class:`~prompt_toolkit.clipboard.InMemoryClipboard`)
    :param rprompt: Text or formatted text to be displayed on the right side.
        This can also be a callable that returns (formatted) text.
    :param bottom_toolbar: Formatted text or callable which is supposed to
        return formatted text.
    :param prompt_continuation: Text that needs to be displayed for a multiline
        prompt continuation. This can either be formatted text or a callable
        that takes a `prompt_width`, `line_number` and `wrap_count` as input
        and returns formatted text. When this is `None` (the default), then
        `prompt_width` spaces will be used.
    :param complete_style: ``CompleteStyle.COLUMN``,
        ``CompleteStyle.MULTI_COLUMN`` or ``CompleteStyle.READLINE_LIKE``.
    :param mouse_support: `bool` or :class:`~prompt_toolkit.filters.Filter`
        to enable mouse support.
    :param placeholder: Text to be displayed when no input has been given
        yet. Unlike the `default` parameter, this won't be returned as part of
        the output ever. This can be formatted text or a callable that returns
        formatted text.
    :param refresh_interval: (number; in seconds) When given, refresh the UI
        every so many seconds.
    :param input: `Input` object. (Note that the preferred way to change the
        input/output is by creating an `AppSession`.)
    :param output: `Output` object.
    """
    _fields = ('message', 'lexer', 'completer', 'complete_in_thread', 'is_password', 'editing_mode', 'key_bindings', 'is_password', 'bottom_toolbar', 'style', 'style_transformation', 'swap_light_and_dark_colors', 'color_depth', 'cursor', 'include_default_pygments_style', 'rprompt', 'multiline', 'prompt_continuation', 'wrap_lines', 'enable_history_search', 'search_ignore_case', 'complete_while_typing', 'validate_while_typing', 'complete_style', 'mouse_support', 'auto_suggest', 'clipboard', 'validator', 'refresh_interval', 'input_processors', 'placeholder', 'enable_system_prompt', 'enable_suspend', 'enable_open_in_editor', 'reserve_space_for_menu', 'tempfile_suffix', 'tempfile')

    def __init__(self, message: AnyFormattedText='', *, multiline: FilterOrBool=False, wrap_lines: FilterOrBool=True, is_password: FilterOrBool=False, vi_mode: bool=False, editing_mode: EditingMode=EditingMode.EMACS, complete_while_typing: FilterOrBool=True, validate_while_typing: FilterOrBool=True, enable_history_search: FilterOrBool=False, search_ignore_case: FilterOrBool=False, lexer: Lexer | None=None, enable_system_prompt: FilterOrBool=False, enable_suspend: FilterOrBool=False, enable_open_in_editor: FilterOrBool=False, validator: Validator | None=None, completer: Completer | None=None, complete_in_thread: bool=False, reserve_space_for_menu: int=8, complete_style: CompleteStyle=CompleteStyle.COLUMN, auto_suggest: AutoSuggest | None=None, style: BaseStyle | None=None, style_transformation: StyleTransformation | None=None, swap_light_and_dark_colors: FilterOrBool=False, color_depth: ColorDepth | None=None, cursor: AnyCursorShapeConfig=None, include_default_pygments_style: FilterOrBool=True, history: History | None=None, clipboard: Clipboard | None=None, prompt_continuation: PromptContinuationText | None=None, rprompt: AnyFormattedText=None, bottom_toolbar: AnyFormattedText=None, mouse_support: FilterOrBool=False, input_processors: list[Processor] | None=None, placeholder: AnyFormattedText | None=None, key_bindings: KeyBindingsBase | None=None, erase_when_done: bool=False, tempfile_suffix: str | Callable[[], str] | None='.txt', tempfile: str | Callable[[], str] | None=None, refresh_interval: float=0, input: Input | None=None, output: Output | None=None) -> None:
        history = history or InMemoryHistory()
        clipboard = clipboard or InMemoryClipboard()
        if vi_mode:
            editing_mode = EditingMode.VI
        self._input = input
        self._output = output
        self.message = message
        self.lexer = lexer
        self.completer = completer
        self.complete_in_thread = complete_in_thread
        self.is_password = is_password
        self.key_bindings = key_bindings
        self.bottom_toolbar = bottom_toolbar
        self.style = style
        self.style_transformation = style_transformation
        self.swap_light_and_dark_colors = swap_light_and_dark_colors
        self.color_depth = color_depth
        self.cursor = cursor
        self.include_default_pygments_style = include_default_pygments_style
        self.rprompt = rprompt
        self.multiline = multiline
        self.prompt_continuation = prompt_continuation
        self.wrap_lines = wrap_lines
        self.enable_history_search = enable_history_search
        self.search_ignore_case = search_ignore_case
        self.complete_while_typing = complete_while_typing
        self.validate_while_typing = validate_while_typing
        self.complete_style = complete_style
        self.mouse_support = mouse_support
        self.auto_suggest = auto_suggest
        self.clipboard = clipboard
        self.validator = validator
        self.refresh_interval = refresh_interval
        self.input_processors = input_processors
        self.placeholder = placeholder
        self.enable_system_prompt = enable_system_prompt
        self.enable_suspend = enable_suspend
        self.enable_open_in_editor = enable_open_in_editor
        self.reserve_space_for_menu = reserve_space_for_menu
        self.tempfile_suffix = tempfile_suffix
        self.tempfile = tempfile
        self.history = history
        self.default_buffer = self._create_default_buffer()
        self.search_buffer = self._create_search_buffer()
        self.layout = self._create_layout()
        self.app = self._create_application(editing_mode, erase_when_done)

    def _dyncond(self, attr_name: str) -> Condition:
        """
        Dynamically take this setting from this 'PromptSession' class.
        `attr_name` represents an attribute name of this class. Its value
        can either be a boolean or a `Filter`.

        This returns something that can be used as either a `Filter`
        or `Filter`.
        """

        @Condition
        def dynamic() -> bool:
            value = cast(FilterOrBool, getattr(self, attr_name))
            return to_filter(value)()
        return dynamic

    def _create_default_buffer(self) -> Buffer:
        """
        Create and return the default input buffer.
        """
        dyncond = self._dyncond

        def accept(buff: Buffer) -> bool:
            """Accept the content of the default buffer. This is called when
            the validation succeeds."""
            cast(Application[str], get_app()).exit(result=buff.document.text)
            return True
        return Buffer(name=DEFAULT_BUFFER, complete_while_typing=Condition(lambda: is_true(self.complete_while_typing) and (not is_true(self.enable_history_search)) and (not self.complete_style == CompleteStyle.READLINE_LIKE)), validate_while_typing=dyncond('validate_while_typing'), enable_history_search=dyncond('enable_history_search'), validator=DynamicValidator(lambda: self.validator), completer=DynamicCompleter(lambda: ThreadedCompleter(self.completer) if self.complete_in_thread and self.completer else self.completer), history=self.history, auto_suggest=DynamicAutoSuggest(lambda: self.auto_suggest), accept_handler=accept, tempfile_suffix=lambda: to_str(self.tempfile_suffix or ''), tempfile=lambda: to_str(self.tempfile or ''))

    def _create_search_buffer(self) -> Buffer:
        return Buffer(name=SEARCH_BUFFER)

    def _create_layout(self) -> Layout:
        """
        Create `Layout` for this prompt.
        """
        dyncond = self._dyncond
        has_before_fragments, get_prompt_text_1, get_prompt_text_2 = _split_multiline_prompt(self._get_prompt)
        default_buffer = self.default_buffer
        search_buffer = self.search_buffer

        @Condition
        def display_placeholder() -> bool:
            return self.placeholder is not None and self.default_buffer.text == ''
        all_input_processors = [HighlightIncrementalSearchProcessor(), HighlightSelectionProcessor(), ConditionalProcessor(AppendAutoSuggestion(), has_focus(default_buffer) & ~is_done), ConditionalProcessor(PasswordProcessor(), dyncond('is_password')), DisplayMultipleCursors(), DynamicProcessor(lambda: merge_processors(self.input_processors or [])), ConditionalProcessor(AfterInput(lambda: self.placeholder), filter=display_placeholder)]
        bottom_toolbar = ConditionalContainer(Window(FormattedTextControl(lambda: self.bottom_toolbar, style='class:bottom-toolbar.text'), style='class:bottom-toolbar', dont_extend_height=True, height=Dimension(min=1)), filter=Condition(lambda: self.bottom_toolbar is not None) & ~is_done & renderer_height_is_known)
        search_toolbar = SearchToolbar(search_buffer, ignore_case=dyncond('search_ignore_case'))
        search_buffer_control = SearchBufferControl(buffer=search_buffer, input_processors=[ReverseSearchProcessor()], ignore_case=dyncond('search_ignore_case'))
        system_toolbar = SystemToolbar(enable_global_bindings=dyncond('enable_system_prompt'))

        def get_search_buffer_control() -> SearchBufferControl:
            """Return the UIControl to be focused when searching start."""
            if is_true(self.multiline):
                return search_toolbar.control
            else:
                return search_buffer_control
        default_buffer_control = BufferControl(buffer=default_buffer, search_buffer_control=get_search_buffer_control, input_processors=all_input_processors, include_default_input_processors=False, lexer=DynamicLexer(lambda: self.lexer), preview_search=True)
        default_buffer_window = Window(default_buffer_control, height=self._get_default_buffer_control_height, get_line_prefix=partial(self._get_line_prefix, get_prompt_text_2=get_prompt_text_2), wrap_lines=dyncond('wrap_lines'))

        @Condition
        def multi_column_complete_style() -> bool:
            return self.complete_style == CompleteStyle.MULTI_COLUMN
        layout = HSplit([FloatContainer(HSplit([ConditionalContainer(Window(FormattedTextControl(get_prompt_text_1), dont_extend_height=True), Condition(has_before_fragments)), ConditionalContainer(default_buffer_window, Condition(lambda: get_app().layout.current_control != search_buffer_control)), ConditionalContainer(Window(search_buffer_control), Condition(lambda: get_app().layout.current_control == search_buffer_control))]), [Float(xcursor=True, ycursor=True, transparent=True, content=CompletionsMenu(max_height=16, scroll_offset=1, extra_filter=has_focus(default_buffer) & ~multi_column_complete_style)), Float(xcursor=True, ycursor=True, transparent=True, content=MultiColumnCompletionsMenu(show_meta=True, extra_filter=has_focus(default_buffer) & multi_column_complete_style)), Float(right=0, top=0, hide_when_covering_content=True, content=_RPrompt(lambda: self.rprompt))]), ConditionalContainer(ValidationToolbar(), filter=~is_done), ConditionalContainer(system_toolbar, dyncond('enable_system_prompt') & ~is_done), ConditionalContainer(Window(FormattedTextControl(self._get_arg_text), height=1), dyncond('multiline') & has_arg), ConditionalContainer(search_toolbar, dyncond('multiline') & ~is_done), bottom_toolbar])
        return Layout(layout, default_buffer_window)

    def _create_application(self, editing_mode: EditingMode, erase_when_done: bool) -> Application[_T]:
        """
        Create the `Application` object.
        """
        dyncond = self._dyncond
        auto_suggest_bindings = load_auto_suggest_bindings()
        open_in_editor_bindings = load_open_in_editor_bindings()
        prompt_bindings = self._create_prompt_bindings()
        application: Application[_T] = Application(layout=self.layout, style=DynamicStyle(lambda: self.style), style_transformation=merge_style_transformations([DynamicStyleTransformation(lambda: self.style_transformation), ConditionalStyleTransformation(SwapLightAndDarkStyleTransformation(), dyncond('swap_light_and_dark_colors'))]), include_default_pygments_style=dyncond('include_default_pygments_style'), clipboard=DynamicClipboard(lambda: self.clipboard), key_bindings=merge_key_bindings([merge_key_bindings([auto_suggest_bindings, ConditionalKeyBindings(open_in_editor_bindings, dyncond('enable_open_in_editor') & has_focus(DEFAULT_BUFFER)), prompt_bindings]), DynamicKeyBindings(lambda: self.key_bindings)]), mouse_support=dyncond('mouse_support'), editing_mode=editing_mode, erase_when_done=erase_when_done, reverse_vi_search_direction=True, color_depth=lambda: self.color_depth, cursor=DynamicCursorShapeConfig(lambda: self.cursor), refresh_interval=self.refresh_interval, input=self._input, output=self._output)
        '\n        def on_render(app):\n            multiline = is_true(self.multiline)\n            current_control = app.layout.current_control\n\n            if multiline:\n                if current_control == search_buffer_control:\n                    app.layout.current_control = search_toolbar.control\n                    app.invalidate()\n            else:\n                if current_control == search_toolbar.control:\n                    app.layout.current_control = search_buffer_control\n                    app.invalidate()\n\n        app.on_render += on_render\n        '
        return application

    def _create_prompt_bindings(self) -> KeyBindings:
        """
        Create the KeyBindings for a prompt application.
        """
        kb = KeyBindings()
        handle = kb.add
        default_focused = has_focus(DEFAULT_BUFFER)

        @Condition
        def do_accept() -> bool:
            return not is_true(self.multiline) and self.app.layout.has_focus(DEFAULT_BUFFER)

        @handle('enter', filter=do_accept & default_focused)
        def _accept_input(event: E) -> None:
            """Accept input when enter has been pressed."""
            self.default_buffer.validate_and_handle()

        @Condition
        def readline_complete_style() -> bool:
            return self.complete_style == CompleteStyle.READLINE_LIKE

        @handle('tab', filter=readline_complete_style & default_focused)
        def _complete_like_readline(event: E) -> None:
            """Display completions (like Readline)."""
            display_completions_like_readline(event)

        @handle('c-c', filter=default_focused)
        @handle('<sigint>')
        def _keyboard_interrupt(event: E) -> None:
            """Abort when Control-C has been pressed."""
            event.app.exit(exception=KeyboardInterrupt, style='class:aborting')

        @Condition
        def ctrl_d_condition() -> bool:
            """Ctrl-D binding is only active when the default buffer is selected
            and empty."""
            app = get_app()
            return app.current_buffer.name == DEFAULT_BUFFER and (not app.current_buffer.text)

        @handle('c-d', filter=ctrl_d_condition & default_focused)
        def _eof(event: E) -> None:
            """Exit when Control-D has been pressed."""
            event.app.exit(exception=EOFError, style='class:exiting')
        suspend_supported = Condition(suspend_to_background_supported)

        @Condition
        def enable_suspend() -> bool:
            return to_filter(self.enable_suspend)()

        @handle('c-z', filter=suspend_supported & enable_suspend)
        def _suspend(event: E) -> None:
            """
            Suspend process to background.
            """
            event.app.suspend_to_background()
        return kb

    def prompt(self, message: AnyFormattedText | None=None, *, editing_mode: EditingMode | None=None, refresh_interval: float | None=None, vi_mode: bool | None=None, lexer: Lexer | None=None, completer: Completer | None=None, complete_in_thread: bool | None=None, is_password: bool | None=None, key_bindings: KeyBindingsBase | None=None, bottom_toolbar: AnyFormattedText | None=None, style: BaseStyle | None=None, color_depth: ColorDepth | None=None, cursor: AnyCursorShapeConfig | None=None, include_default_pygments_style: FilterOrBool | None=None, style_transformation: StyleTransformation | None=None, swap_light_and_dark_colors: FilterOrBool | None=None, rprompt: AnyFormattedText | None=None, multiline: FilterOrBool | None=None, prompt_continuation: PromptContinuationText | None=None, wrap_lines: FilterOrBool | None=None, enable_history_search: FilterOrBool | None=None, search_ignore_case: FilterOrBool | None=None, complete_while_typing: FilterOrBool | None=None, validate_while_typing: FilterOrBool | None=None, complete_style: CompleteStyle | None=None, auto_suggest: AutoSuggest | None=None, validator: Validator | None=None, clipboard: Clipboard | None=None, mouse_support: FilterOrBool | None=None, input_processors: list[Processor] | None=None, placeholder: AnyFormattedText | None=None, reserve_space_for_menu: int | None=None, enable_system_prompt: FilterOrBool | None=None, enable_suspend: FilterOrBool | None=None, enable_open_in_editor: FilterOrBool | None=None, tempfile_suffix: str | Callable[[], str] | None=None, tempfile: str | Callable[[], str] | None=None, default: str | Document='', accept_default: bool=False, pre_run: Callable[[], None] | None=None, set_exception_handler: bool=True, handle_sigint: bool=True, in_thread: bool=False, inputhook: InputHook | None=None) -> _T:
        """
        Display the prompt.

        The first set of arguments is a subset of the :class:`~.PromptSession`
        class itself. For these, passing in ``None`` will keep the current
        values that are active in the session. Passing in a value will set the
        attribute for the session, which means that it applies to the current,
        but also to the next prompts.

        Note that in order to erase a ``Completer``, ``Validator`` or
        ``AutoSuggest``, you can't use ``None``. Instead pass in a
        ``DummyCompleter``, ``DummyValidator`` or ``DummyAutoSuggest`` instance
        respectively. For a ``Lexer`` you can pass in an empty ``SimpleLexer``.

        Additional arguments, specific for this prompt:

        :param default: The default input text to be shown. (This can be edited
            by the user).
        :param accept_default: When `True`, automatically accept the default
            value without allowing the user to edit the input.
        :param pre_run: Callable, called at the start of `Application.run`.
        :param in_thread: Run the prompt in a background thread; block the
            current thread. This avoids interference with an event loop in the
            current thread. Like `Application.run(in_thread=True)`.

        This method will raise ``KeyboardInterrupt`` when control-c has been
        pressed (for abort) and ``EOFError`` when control-d has been pressed
        (for exit).
        """
        if message is not None:
            self.message = message
        if editing_mode is not None:
            self.editing_mode = editing_mode
        if refresh_interval is not None:
            self.refresh_interval = refresh_interval
        if vi_mode:
            self.editing_mode = EditingMode.VI
        if lexer is not None:
            self.lexer = lexer
        if completer is not None:
            self.completer = completer
        if complete_in_thread is not None:
            self.complete_in_thread = complete_in_thread
        if is_password is not None:
            self.is_password = is_password
        if key_bindings is not None:
            self.key_bindings = key_bindings
        if bottom_toolbar is not None:
            self.bottom_toolbar = bottom_toolbar
        if style is not None:
            self.style = style
        if color_depth is not None:
            self.color_depth = color_depth
        if cursor is not None:
            self.cursor = cursor
        if include_default_pygments_style is not None:
            self.include_default_pygments_style = include_default_pygments_style
        if style_transformation is not None:
            self.style_transformation = style_transformation
        if swap_light_and_dark_colors is not None:
            self.swap_light_and_dark_colors = swap_light_and_dark_colors
        if rprompt is not None:
            self.rprompt = rprompt
        if multiline is not None:
            self.multiline = multiline
        if prompt_continuation is not None:
            self.prompt_continuation = prompt_continuation
        if wrap_lines is not None:
            self.wrap_lines = wrap_lines
        if enable_history_search is not None:
            self.enable_history_search = enable_history_search
        if search_ignore_case is not None:
            self.search_ignore_case = search_ignore_case
        if complete_while_typing is not None:
            self.complete_while_typing = complete_while_typing
        if validate_while_typing is not None:
            self.validate_while_typing = validate_while_typing
        if complete_style is not None:
            self.complete_style = complete_style
        if auto_suggest is not None:
            self.auto_suggest = auto_suggest
        if validator is not None:
            self.validator = validator
        if clipboard is not None:
            self.clipboard = clipboard
        if mouse_support is not None:
            self.mouse_support = mouse_support
        if input_processors is not None:
            self.input_processors = input_processors
        if placeholder is not None:
            self.placeholder = placeholder
        if reserve_space_for_menu is not None:
            self.reserve_space_for_menu = reserve_space_for_menu
        if enable_system_prompt is not None:
            self.enable_system_prompt = enable_system_prompt
        if enable_suspend is not None:
            self.enable_suspend = enable_suspend
        if enable_open_in_editor is not None:
            self.enable_open_in_editor = enable_open_in_editor
        if tempfile_suffix is not None:
            self.tempfile_suffix = tempfile_suffix
        if tempfile is not None:
            self.tempfile = tempfile
        self._add_pre_run_callables(pre_run, accept_default)
        self.default_buffer.reset(default if isinstance(default, Document) else Document(default))
        self.app.refresh_interval = self.refresh_interval
        if self._output is None and is_dumb_terminal():
            with self._dumb_prompt(self.message) as dump_app:
                return dump_app.run(in_thread=in_thread, handle_sigint=handle_sigint)
        return self.app.run(set_exception_handler=set_exception_handler, in_thread=in_thread, handle_sigint=handle_sigint, inputhook=inputhook)

    @contextmanager
    def _dumb_prompt(self, message: AnyFormattedText='') -> Iterator[Application[_T]]:
        """
        Create prompt `Application` for prompt function for dumb terminals.

        Dumb terminals have minimum rendering capabilities. We can only print
        text to the screen. We can't use colors, and we can't do cursor
        movements. The Emacs inferior shell is an example of a dumb terminal.

        We will show the prompt, and wait for the input. We still handle arrow
        keys, and all custom key bindings, but we don't really render the
        cursor movements. Instead we only print the typed character that's
        right before the cursor.
        """
        self.output.write(fragment_list_to_text(to_formatted_text(self.message)))
        self.output.flush()
        key_bindings: KeyBindingsBase = self._create_prompt_bindings()
        if self.key_bindings:
            key_bindings = merge_key_bindings([self.key_bindings, key_bindings])
        application = cast(Application[_T], Application(input=self.input, output=DummyOutput(), layout=self.layout, key_bindings=key_bindings))

        def on_text_changed(_: object) -> None:
            self.output.write(self.default_buffer.document.text_before_cursor[-1:])
            self.output.flush()
        self.default_buffer.on_text_changed += on_text_changed
        try:
            yield application
        finally:
            self.output.write('\r\n')
            self.output.flush()
            self.default_buffer.on_text_changed -= on_text_changed

    async def prompt_async(self, message: AnyFormattedText | None=None, *, editing_mode: EditingMode | None=None, refresh_interval: float | None=None, vi_mode: bool | None=None, lexer: Lexer | None=None, completer: Completer | None=None, complete_in_thread: bool | None=None, is_password: bool | None=None, key_bindings: KeyBindingsBase | None=None, bottom_toolbar: AnyFormattedText | None=None, style: BaseStyle | None=None, color_depth: ColorDepth | None=None, cursor: CursorShapeConfig | None=None, include_default_pygments_style: FilterOrBool | None=None, style_transformation: StyleTransformation | None=None, swap_light_and_dark_colors: FilterOrBool | None=None, rprompt: AnyFormattedText | None=None, multiline: FilterOrBool | None=None, prompt_continuation: PromptContinuationText | None=None, wrap_lines: FilterOrBool | None=None, enable_history_search: FilterOrBool | None=None, search_ignore_case: FilterOrBool | None=None, complete_while_typing: FilterOrBool | None=None, validate_while_typing: FilterOrBool | None=None, complete_style: CompleteStyle | None=None, auto_suggest: AutoSuggest | None=None, validator: Validator | None=None, clipboard: Clipboard | None=None, mouse_support: FilterOrBool | None=None, input_processors: list[Processor] | None=None, placeholder: AnyFormattedText | None=None, reserve_space_for_menu: int | None=None, enable_system_prompt: FilterOrBool | None=None, enable_suspend: FilterOrBool | None=None, enable_open_in_editor: FilterOrBool | None=None, tempfile_suffix: str | Callable[[], str] | None=None, tempfile: str | Callable[[], str] | None=None, default: str | Document='', accept_default: bool=False, pre_run: Callable[[], None] | None=None, set_exception_handler: bool=True, handle_sigint: bool=True) -> _T:
        if message is not None:
            self.message = message
        if editing_mode is not None:
            self.editing_mode = editing_mode
        if refresh_interval is not None:
            self.refresh_interval = refresh_interval
        if vi_mode:
            self.editing_mode = EditingMode.VI
        if lexer is not None:
            self.lexer = lexer
        if completer is not None:
            self.completer = completer
        if complete_in_thread is not None:
            self.complete_in_thread = complete_in_thread
        if is_password is not None:
            self.is_password = is_password
        if key_bindings is not None:
            self.key_bindings = key_bindings
        if bottom_toolbar is not None:
            self.bottom_toolbar = bottom_toolbar
        if style is not None:
            self.style = style
        if color_depth is not None:
            self.color_depth = color_depth
        if cursor is not None:
            self.cursor = cursor
        if include_default_pygments_style is not None:
            self.include_default_pygments_style = include_default_pygments_style
        if style_transformation is not None:
            self.style_transformation = style_transformation
        if swap_light_and_dark_colors is not None:
            self.swap_light_and_dark_colors = swap_light_and_dark_colors
        if rprompt is not None:
            self.rprompt = rprompt
        if multiline is not None:
            self.multiline = multiline
        if prompt_continuation is not None:
            self.prompt_continuation = prompt_continuation
        if wrap_lines is not None:
            self.wrap_lines = wrap_lines
        if enable_history_search is not None:
            self.enable_history_search = enable_history_search
        if search_ignore_case is not None:
            self.search_ignore_case = search_ignore_case
        if complete_while_typing is not None:
            self.complete_while_typing = complete_while_typing
        if validate_while_typing is not None:
            self.validate_while_typing = validate_while_typing
        if complete_style is not None:
            self.complete_style = complete_style
        if auto_suggest is not None:
            self.auto_suggest = auto_suggest
        if validator is not None:
            self.validator = validator
        if clipboard is not None:
            self.clipboard = clipboard
        if mouse_support is not None:
            self.mouse_support = mouse_support
        if input_processors is not None:
            self.input_processors = input_processors
        if placeholder is not None:
            self.placeholder = placeholder
        if reserve_space_for_menu is not None:
            self.reserve_space_for_menu = reserve_space_for_menu
        if enable_system_prompt is not None:
            self.enable_system_prompt = enable_system_prompt
        if enable_suspend is not None:
            self.enable_suspend = enable_suspend
        if enable_open_in_editor is not None:
            self.enable_open_in_editor = enable_open_in_editor
        if tempfile_suffix is not None:
            self.tempfile_suffix = tempfile_suffix
        if tempfile is not None:
            self.tempfile = tempfile
        self._add_pre_run_callables(pre_run, accept_default)
        self.default_buffer.reset(default if isinstance(default, Document) else Document(default))
        self.app.refresh_interval = self.refresh_interval
        if self._output is None and is_dumb_terminal():
            with self._dumb_prompt(self.message) as dump_app:
                return await dump_app.run_async(handle_sigint=handle_sigint)
        return await self.app.run_async(set_exception_handler=set_exception_handler, handle_sigint=handle_sigint)

    def _add_pre_run_callables(self, pre_run: Callable[[], None] | None, accept_default: bool) -> None:

        def pre_run2() -> None:
            if pre_run:
                pre_run()
            if accept_default:
                get_running_loop().call_soon(self.default_buffer.validate_and_handle)
        self.app.pre_run_callables.append(pre_run2)

    @property
    def editing_mode(self) -> EditingMode:
        return self.app.editing_mode

    @editing_mode.setter
    def editing_mode(self, value: EditingMode) -> None:
        self.app.editing_mode = value

    def _get_default_buffer_control_height(self) -> Dimension:
        if self.completer is not None and self.complete_style != CompleteStyle.READLINE_LIKE:
            space = self.reserve_space_for_menu
        else:
            space = 0
        if space and (not get_app().is_done):
            buff = self.default_buffer
            if buff.complete_while_typing() or buff.complete_state is not None:
                return Dimension(min=space)
        return Dimension()

    def _get_prompt(self) -> StyleAndTextTuples:
        return to_formatted_text(self.message, style='class:prompt')

    def _get_continuation(self, width: int, line_number: int, wrap_count: int) -> StyleAndTextTuples:
        """
        Insert the prompt continuation.

        :param width: The width that was used for the prompt. (more or less can
            be used.)
        :param line_number:
        :param wrap_count: Amount of times that the line has been wrapped.
        """
        prompt_continuation = self.prompt_continuation
        if callable(prompt_continuation):
            continuation: AnyFormattedText = prompt_continuation(width, line_number, wrap_count)
        else:
            continuation = prompt_continuation
        if continuation is None and is_true(self.multiline):
            continuation = ' ' * width
        return to_formatted_text(continuation, style='class:prompt-continuation')

    def _get_line_prefix(self, line_number: int, wrap_count: int, get_prompt_text_2: _StyleAndTextTuplesCallable) -> StyleAndTextTuples:
        """
        Return whatever needs to be inserted before every line.
        (the prompt, or a line continuation.)
        """
        if line_number == 0 and wrap_count == 0:
            if not is_true(self.multiline) and get_app().key_processor.arg is not None:
                return self._inline_arg()
            else:
                return get_prompt_text_2()
        prompt_width = get_cwidth(fragment_list_to_text(get_prompt_text_2()))
        return self._get_continuation(prompt_width, line_number, wrap_count)

    def _get_arg_text(self) -> StyleAndTextTuples:
        """'arg' toolbar, for in multiline mode."""
        arg = self.app.key_processor.arg
        if arg is None:
            return []
        if arg == '-':
            arg = '-1'
        return [('class:arg-toolbar', 'Repeat: '), ('class:arg-toolbar.text', arg)]

    def _inline_arg(self) -> StyleAndTextTuples:
        """'arg' prefix, for in single line mode."""
        app = get_app()
        if app.key_processor.arg is None:
            return []
        else:
            arg = app.key_processor.arg
            return [('class:prompt.arg', '(arg: '), ('class:prompt.arg.text', str(arg)), ('class:prompt.arg', ') ')]

    @property
    def input(self) -> Input:
        return self.app.input

    @property
    def output(self) -> Output:
        return self.app.output