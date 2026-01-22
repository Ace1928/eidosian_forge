from __future__ import annotations
from warnings import warn
import ast
import codeop
import io
import re
import sys
import tokenize
import warnings
from typing import List, Tuple, Union, Optional, TYPE_CHECKING
from types import CodeType
from IPython.core.inputtransformer import (leading_indent,
from IPython.utils import tokenutil
from IPython.core.inputtransformer import (ESC_SHELL, ESC_SH_CAP, ESC_HELP,
class IncompleteString:
    type = exact_type = INCOMPLETE_STRING

    def __init__(self, s, start, end, line):
        self.s = s
        self.start = start
        self.end = end
        self.line = line