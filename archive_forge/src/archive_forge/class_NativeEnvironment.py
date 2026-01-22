import typing as t
from ast import literal_eval
from ast import parse
from itertools import chain
from itertools import islice
from types import GeneratorType
from . import nodes
from .compiler import CodeGenerator
from .compiler import Frame
from .compiler import has_safe_repr
from .environment import Environment
from .environment import Template
class NativeEnvironment(Environment):
    """An environment that renders templates to native Python types."""
    code_generator_class = NativeCodeGenerator
    concat = staticmethod(native_concat)