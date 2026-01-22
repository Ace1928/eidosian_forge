from __future__ import annotations
from typing import TYPE_CHECKING, cast
from streamlit.proto.Text_pb2 import Text as TextProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.string_util import clean_text
class TextMixin:

    @gather_metrics('text')
    def text(self, body: SupportsStr, *, help: str | None=None) -> DeltaGenerator:
        """Write fixed-width and preformatted text.

        Parameters
        ----------
        body : str
            The string to display.

        help : str
            An optional tooltip that gets displayed next to the text.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> st.text('This is some text.')

        """
        text_proto = TextProto()
        text_proto.body = clean_text(body)
        if help:
            text_proto.help = help
        return self.dg._enqueue('text', text_proto)

    @property
    def dg(self) -> DeltaGenerator:
        """Get our DeltaGenerator."""
        return cast('DeltaGenerator', self)