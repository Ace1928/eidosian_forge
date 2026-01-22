from dataclasses import dataclass
from inspect import signature
from textwrap import dedent
import ast
import html
import inspect
import io as stdlib_io
import linecache
import os
import types
import warnings
from typing import (
import traitlets
from IPython.core import page
from IPython.lib.pretty import pretty
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils import PyColorize, openpy
from IPython.utils.dir2 import safe_hasattr
from IPython.utils.path import compress_user
from IPython.utils.text import indent
from IPython.utils.wildcard import list_namespace, typestr2type
from IPython.utils.coloransi import TermColors
from IPython.utils.colorable import Colorable
from IPython.utils.decorators import undoc
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
def _render_signature(obj_signature, obj_name) -> str:
    """
    This was mostly taken from inspect.Signature.__str__.
    Look there for the comments.
    The only change is to add linebreaks when this gets too long.
    """
    result = []
    pos_only = False
    kw_only = True
    for param in obj_signature.parameters.values():
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            pos_only = True
        elif pos_only:
            result.append('/')
            pos_only = False
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            kw_only = False
        elif param.kind == inspect.Parameter.KEYWORD_ONLY and kw_only:
            result.append('*')
            kw_only = False
        result.append(str(param))
    if pos_only:
        result.append('/')
    if len(obj_name) + sum((len(r) + 2 for r in result)) > 75:
        rendered = '{}(\n{})'.format(obj_name, ''.join(('    {},\n'.format(r) for r in result)))
    else:
        rendered = '{}({})'.format(obj_name, ', '.join(result))
    if obj_signature.return_annotation is not inspect._empty:
        anno = inspect.formatannotation(obj_signature.return_annotation)
        rendered += ' -> {}'.format(anno)
    return rendered