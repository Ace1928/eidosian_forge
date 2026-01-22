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
def _format_docutils_params(self, fields: List[Tuple[str, str, List[str]]], field_role: str='param', type_role: str='type') -> List[str]:
    lines = []
    for _name, _type, _desc in fields:
        _desc = self._strip_empty(_desc)
        if any(_desc):
            _desc = self._fix_field_desc(_desc)
            field = ':%s %s: ' % (field_role, _name)
            lines.extend(self._format_block(field, _desc))
        else:
            lines.append(':%s %s:' % (field_role, _name))
        if _type:
            lines.append(':%s %s: %s' % (type_role, _name, _type))
    return lines + ['']