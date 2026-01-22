import os
import re
import warnings
from contextlib import contextmanager
from copy import copy
from os import path
from types import ModuleType
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Set,
import docutils
from docutils import nodes
from docutils.io import FileOutput
from docutils.nodes import Element, Node, system_message
from docutils.parsers.rst import Directive, directives, roles
from docutils.parsers.rst.states import Inliner
from docutils.statemachine import State, StateMachine, StringList
from docutils.utils import Reporter, unescape
from docutils.writers._html_base import HTMLTranslator
from sphinx.deprecation import RemovedInSphinx70Warning, deprecated_alias
from sphinx.errors import SphinxError
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.typing import RoleFunction
class SphinxFileOutput(FileOutput):
    """Better FileOutput class for Sphinx."""

    def __init__(self, **kwargs: Any) -> None:
        self.overwrite_if_changed = kwargs.pop('overwrite_if_changed', False)
        kwargs.setdefault('encoding', 'utf-8')
        super().__init__(**kwargs)

    def write(self, data: str) -> str:
        if self.destination_path and self.autoclose and ('b' not in self.mode) and self.overwrite_if_changed and os.path.exists(self.destination_path):
            with open(self.destination_path, encoding=self.encoding) as f:
                if f.read() == data:
                    return data
        return super().write(data)