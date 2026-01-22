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
def _transform_line(self, line):
    """Push a line of input code through the various transformers.

        Returns any output from the transformers, or None if a transformer
        is accumulating lines.

        Sets self.transformer_accumulating as a side effect.
        """

    def _accumulating(dbg):
        self.transformer_accumulating = True
        return None
    for transformer in self.physical_line_transforms:
        line = transformer.push(line)
        if line is None:
            return _accumulating(transformer)
    if not self.within_python_line:
        line = self.assemble_logical_lines.push(line)
        if line is None:
            return _accumulating('acc logical line')
        for transformer in self.logical_line_transforms:
            line = transformer.push(line)
            if line is None:
                return _accumulating(transformer)
    line = self.assemble_python_lines.push(line)
    if line is None:
        self.within_python_line = True
        return _accumulating('acc python line')
    else:
        self.within_python_line = False
    for transformer in self.python_line_transforms:
        line = transformer.push(line)
        if line is None:
            return _accumulating(transformer)
    self.transformer_accumulating = False
    return line