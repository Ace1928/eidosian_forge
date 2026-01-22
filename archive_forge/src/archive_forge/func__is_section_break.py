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
def _is_section_break(self) -> bool:
    line1, line2 = (self._lines.get(0), self._lines.get(1))
    return not self._lines or self._is_section_header() or ['', ''] == [line1, line2] or (self._is_in_section and line1 and (not self._is_indented(line1, self._section_indent)))