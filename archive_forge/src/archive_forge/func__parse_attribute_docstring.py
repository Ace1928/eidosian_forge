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
def _parse_attribute_docstring(self) -> List[str]:
    _type, _desc = self._consume_inline_attribute()
    lines = self._format_field('', '', _desc)
    if _type:
        lines.extend(['', ':type: %s' % _type])
    return lines