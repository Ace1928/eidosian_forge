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
def _format_fields(self, field_type: str, fields: List[Tuple[str, str, List[str]]]) -> List[str]:
    field_type = ':%s:' % field_type.strip()
    padding = ' ' * len(field_type)
    multi = len(fields) > 1
    lines: List[str] = []
    for _name, _type, _desc in fields:
        field = self._format_field(_name, _type, _desc)
        if multi:
            if lines:
                lines.extend(self._format_block(padding + ' * ', field))
            else:
                lines.extend(self._format_block(field_type + ' * ', field))
        else:
            lines.extend(self._format_block(field_type + ' ', field))
    if lines and lines[-1]:
        lines.append('')
    return lines