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
class UnboundSymbols(EnvTransform, SkipDeclarations):

    def __init__(self):
        super(EnvTransform, self).__init__(context=None)
        self.unbound = set()

    def visit_NameNode(self, node):
        if not self.current_env().lookup(node.name):
            self.unbound.add(node.name)
        return node

    def __call__(self, node):
        super(UnboundSymbols, self).__call__(node)
        return self.unbound