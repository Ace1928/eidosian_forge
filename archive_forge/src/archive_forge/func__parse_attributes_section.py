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
def _parse_attributes_section(self, section: str) -> List[str]:
    lines = []
    for _name, _type, _desc in self._consume_fields():
        if not _type:
            _type = self._lookup_annotation(_name)
        if self._config.napoleon_use_ivar:
            field = ':ivar %s: ' % _name
            lines.extend(self._format_block(field, _desc))
            if _type:
                lines.append(':vartype %s: %s' % (_name, _type))
        else:
            lines.append('.. attribute:: ' + _name)
            if self._opt and 'noindex' in self._opt:
                lines.append('   :noindex:')
            lines.append('')
            fields = self._format_field('', '', _desc)
            lines.extend(self._indent(fields, 3))
            if _type:
                lines.append('')
                lines.extend(self._indent([':type: %s' % _type], 3))
            lines.append('')
    if self._config.napoleon_use_ivar:
        lines.append('')
    return lines