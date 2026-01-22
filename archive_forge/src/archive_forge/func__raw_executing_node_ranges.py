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
def _raw_executing_node_ranges(self, common_indent=0) -> List[RangeInLine]:
    ex = self.frame_info.executing
    node = ex.node
    if node:
        rang = self.range_from_node(node, ex, common_indent)
        if rang:
            return [rang]
    return []