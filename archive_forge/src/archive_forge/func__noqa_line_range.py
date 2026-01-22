from __future__ import annotations
import argparse
import ast
import functools
import logging
import tokenize
from typing import Any
from typing import Generator
from typing import List
from typing import Tuple
from flake8 import defaults
from flake8 import utils
from flake8._compat import FSTRING_END
from flake8._compat import FSTRING_MIDDLE
from flake8.plugins.finder import LoadedPlugin
def _noqa_line_range(self, min_line: int, max_line: int) -> dict[int, str]:
    line_range = range(min_line, max_line + 1)
    joined = ''.join(self.lines[min_line - 1:max_line])
    return dict.fromkeys(line_range, joined)