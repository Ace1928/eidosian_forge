import ast
import html
import os
import sys
from collections import defaultdict, Counter
from enum import Enum
from textwrap import dedent
from types import FrameType, CodeType, TracebackType
from typing import (
from typing import Mapping
import executing
from asttokens.util import Token
from executing import only
from pure_eval import Evaluator, is_expression_interesting
from stack_data.utils import (
@cached_property
def _pygmented_scope_lines(self) -> Optional[Tuple[int, List[str]]]:
    from pygments.formatters import HtmlFormatter
    formatter = self.options.pygments_formatter
    scope = self.scope
    assert_(formatter, ValueError('Must set a pygments formatter in Options'))
    assert_(scope)
    if isinstance(formatter, HtmlFormatter):
        formatter.nowrap = True
    atext = self.source.asttext()
    node = self.executing.node
    if node and getattr(formatter.style, 'for_executing_node', False):
        scope_start = atext.get_text_range(scope)[0]
        start, end = atext.get_text_range(node)
        start -= scope_start
        end -= scope_start
        ranges = [(start, end)]
    else:
        ranges = []
    code = atext.get_text(scope)
    lines = _pygmented_with_ranges(formatter, code, ranges)
    start_line = self.source.line_range(scope)[0]
    return (start_line, lines)