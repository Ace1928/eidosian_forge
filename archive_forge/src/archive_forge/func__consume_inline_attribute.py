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
def _consume_inline_attribute(self) -> Tuple[str, List[str]]:
    line = self._lines.next()
    _type, colon, _desc = self._partition_field_on_colon(line)
    if not colon or not _desc:
        _type, _desc = (_desc, _type)
        _desc += colon
    _descs = [_desc] + self._dedent(self._consume_to_end())
    _descs = self.__class__(_descs, self._config).lines()
    return (_type, _descs)