import inspect
import itertools
import re
import tokenize
from collections import OrderedDict
from inspect import Signature
from token import DEDENT, INDENT, NAME, NEWLINE, NUMBER, OP, STRING
from tokenize import COMMENT, NL
from typing import Any, Dict, List, Optional, Tuple
from sphinx.pycode.ast import ast  # for py37 or older
from sphinx.pycode.ast import parse, unparse
def add_overload_entry(self, func: ast.FunctionDef) -> None:
    from sphinx.util.inspect import signature_from_ast
    qualname = self.get_qualname_for(func.name)
    if qualname:
        overloads = self.overloads.setdefault('.'.join(qualname), [])
        overloads.append(signature_from_ast(func))