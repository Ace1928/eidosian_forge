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
def token_ranges(self) -> List[RangeInLine]:
    """
        A list of RangeInLines for each token in .tokens,
        where range.data is a Token object from asttokens:
        https://asttokens.readthedocs.io/en/latest/api-index.html#asttokens.util.Token
        """
    return [RangeInLine(token.start[1], token.end[1], token) for token in self.tokens]