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
def _consume_fields(self, parse_type: bool=True, prefer_type: bool=False, multiple: bool=False) -> List[Tuple[str, str, List[str]]]:
    self._consume_empty()
    fields = []
    while not self._is_section_break():
        _name, _type, _desc = self._consume_field(parse_type, prefer_type)
        if multiple and _name:
            for name in _name.split(','):
                fields.append((name.strip(), _type, _desc))
        elif _name or _type or _desc:
            fields.append((_name, _type, _desc))
    return fields