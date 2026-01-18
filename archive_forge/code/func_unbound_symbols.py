from __future__ import absolute_import
import hashlib
import inspect
import os
import re
import sys
from distutils.core import Distribution, Extension
from distutils.command.build_ext import build_ext
import Cython
from ..Compiler.Main import Context
from ..Compiler.Options import (default_options, CompilationOptions,
from ..Compiler.Visitor import CythonTransform, EnvTransform
from ..Compiler.ParseTreeTransforms import SkipDeclarations
from ..Compiler.TreeFragment import parse_from_strings
from ..Compiler.StringEncoding import _unicode
from .Dependencies import strip_string_literals, cythonize, cached_function
from ..Compiler import Pipeline
from ..Utils import get_cython_cache_dir
import cython as cython_module
@cached_function
def unbound_symbols(code, context=None):
    code = to_unicode(code)
    if context is None:
        context = Context([], get_directive_defaults(), options=CompilationOptions(default_options))
    from ..Compiler.ParseTreeTransforms import AnalyseDeclarationsTransform
    tree = parse_from_strings('(tree fragment)', code)
    for phase in Pipeline.create_pipeline(context, 'pyx'):
        if phase is None:
            continue
        tree = phase(tree)
        if isinstance(phase, AnalyseDeclarationsTransform):
            break
    try:
        import builtins
    except ImportError:
        import __builtin__ as builtins
    return tuple(UnboundSymbols()(tree) - set(dir(builtins)))