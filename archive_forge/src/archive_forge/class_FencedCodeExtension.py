from __future__ import annotations
from textwrap import dedent
from . import Extension
from ..preprocessors import Preprocessor
from .codehilite import CodeHilite, CodeHiliteExtension, parse_hl_lines
from .attr_list import get_attrs_and_remainder, AttrListExtension
from ..util import parseBoolValue
from ..serializers import _escape_attrib_html
import re
from typing import TYPE_CHECKING, Any, Iterable
class FencedCodeExtension(Extension):

    def __init__(self, **kwargs):
        self.config = {'lang_prefix': ['language-', 'Prefix prepended to the language. Default: "language-"']}
        ' Default configuration options. '
        super().__init__(**kwargs)

    def extendMarkdown(self, md):
        """ Add `FencedBlockPreprocessor` to the Markdown instance. """
        md.registerExtension(self)
        md.preprocessors.register(FencedBlockPreprocessor(md, self.getConfigs()), 'fenced_code_block', 25)