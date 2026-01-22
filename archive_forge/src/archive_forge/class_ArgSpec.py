import inspect
import keyword
import pydoc
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, Dict, List, ContextManager
from types import MemberDescriptorType, TracebackType
from ._typing_compat import Literal
from pygments.token import Token
from pygments.lexers import Python3Lexer
from .lazyre import LazyReCompile
@dataclass
class ArgSpec:
    args: List[str]
    varargs: Optional[str]
    varkwargs: Optional[str]
    defaults: Optional[List[_Repr]]
    kwonly: List[str]
    kwonly_defaults: Optional[Dict[str, _Repr]]
    annotations: Optional[Dict[str, Any]]