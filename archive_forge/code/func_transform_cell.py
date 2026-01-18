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
def transform_cell(self, cell):
    """Process and translate a cell of input.
        """
    self.reset()
    try:
        self.push(cell)
        self.flush_transformers()
        return self.source
    finally:
        self.reset()