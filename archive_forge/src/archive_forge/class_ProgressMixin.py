from __future__ import annotations
import math
from typing import TYPE_CHECKING, Union, cast
from typing_extensions import TypeAlias
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Progress_pb2 import Progress as ProgressProto
from streamlit.string_util import clean_text
class ProgressMixin:

    def progress(self, value: FloatOrInt, text: str | None=None) -> DeltaGenerator:
        """Display a progress bar.

        Parameters
        ----------
        value : int or float
            0 <= value <= 100 for int

            0.0 <= value <= 1.0 for float

        text : str or None
            A message to display above the progress bar. The text can optionally
            contain Markdown and supports the following elements: Bold, Italics,
            Strikethroughs, Inline Code, Emojis, and Links.

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

        Example
        -------
        Here is an example of a progress bar increasing over time and disappearing when it reaches completion:

        >>> import streamlit as st
        >>> import time
        >>>
        >>> progress_text = "Operation in progress. Please wait."
        >>> my_bar = st.progress(0, text=progress_text)
        >>>
        >>> for percent_complete in range(100):
        ...     time.sleep(0.01)
        ...     my_bar.progress(percent_complete + 1, text=progress_text)
        >>> time.sleep(1)
        >>> my_bar.empty()
        >>>
        >>> st.button("Rerun")

        .. output::
           https://doc-status-progress.streamlit.app/
           height: 220px

        """
        progress_proto = ProgressProto()
        progress_proto.value = _get_value(value)
        text = _get_text(text)
        if text is not None:
            progress_proto.text = text
        return self.dg._enqueue('progress', progress_proto)

    @property
    def dg(self) -> DeltaGenerator:
        """Get our DeltaGenerator."""
        return cast('DeltaGenerator', self)