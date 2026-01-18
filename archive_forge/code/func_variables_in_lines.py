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
def variables_in_lines(self) -> List[Variable]:
    """
        A list of Variable objects contained within the lines returned by .lines.
        """
    return unique_in_order((var for line in self.lines if isinstance(line, Line) for var, node in self.variables_by_lineno[line.lineno]))