from __future__ import annotations
import inspect
import re
import sys
import textwrap
import types
from ast import FunctionDef, Module, stmt
from dataclasses import dataclass
from typing import Any, AnyStr, Callable, ForwardRef, NewType, TypeVar, get_type_hints
from docutils.frontend import OptionParser
from docutils.nodes import Node
from docutils.parsers.rst import Parser as RstParser
from docutils.utils import new_document
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc import Options
from sphinx.ext.autodoc.mock import mock
from sphinx.util import logging
from sphinx.util.inspect import signature as sphinx_signature
from sphinx.util.inspect import stringify_signature
from .patches import install_patches
from .version import __version__
def _line_is_param_line_for_arg(line: str, arg_name: str) -> bool:
    """Return True if `line` is a valid parameter line for `arg_name`, false otherwise."""
    keyword_and_name = _get_sphinx_line_keyword_and_argument(line)
    if keyword_and_name is None:
        return False
    keyword, doc_name = keyword_and_name
    if doc_name is None:
        return False
    if keyword not in {'param', 'parameter', 'arg', 'argument'}:
        return False
    for prefix in ('', '\\*', '\\**', '\\*\\*'):
        if doc_name == prefix + arg_name:
            return True
    return False