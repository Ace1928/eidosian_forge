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
def _parse_usage_section(self, section: str) -> List[str]:
    header = ['.. rubric:: Usage:', '']
    block = ['.. code-block:: python', '']
    lines = self._consume_usage_section()
    lines = self._indent(lines, 3)
    return header + block + lines + ['']