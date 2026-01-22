import collections
import inspect
import re
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, Union
from sphinx.application import Sphinx
from sphinx.config import Config as SphinxConfig
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.inspect import stringify_annotation
from sphinx.util.typing import get_type_hints
def _strip_empty(self, lines: List[str]) -> List[str]:
    if lines:
        start = -1
        for i, line in enumerate(lines):
            if line:
                start = i
                break
        if start == -1:
            lines = []
        end = -1
        for i in reversed(range(len(lines))):
            line = lines[i]
            if line:
                end = i
                break
        if start > 0 or end + 1 < len(lines):
            lines = lines[start:end + 1]
    return lines