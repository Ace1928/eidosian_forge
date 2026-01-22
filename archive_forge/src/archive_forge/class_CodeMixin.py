from __future__ import annotations
from typing import TYPE_CHECKING, cast
from streamlit.proto.Code_pb2 import Code as CodeProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.string_util import clean_text
from streamlit.type_util import SupportsStr
class CodeMixin:

    @gather_metrics('code')
    def code(self, body: SupportsStr, language: str | None='python', line_numbers: bool=False) -> DeltaGenerator:
        """Display a code block with optional syntax highlighting.

        Parameters
        ----------
        body : str
            The string to display as code.

        language : str or None
            The language that the code is written in, for syntax highlighting.
            If ``None``, the code will be unstyled. Defaults to ``"python"``.

            For a list of available ``language`` values, see:

            https://github.com/react-syntax-highlighter/react-syntax-highlighter/blob/master/AVAILABLE_LANGUAGES_PRISM.MD

        line_numbers : bool
            An optional boolean indicating whether to show line numbers to the
            left of the code block. Defaults to ``False``.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> code = '''def hello():
        ...     print("Hello, Streamlit!")'''
        >>> st.code(code, language='python')

        """
        code_proto = CodeProto()
        code_proto.code_text = clean_text(body)
        code_proto.language = language or 'plaintext'
        code_proto.show_line_numbers = line_numbers
        return self.dg._enqueue('code', code_proto)

    @property
    def dg(self) -> DeltaGenerator:
        """Get our DeltaGenerator."""
        return cast('DeltaGenerator', self)