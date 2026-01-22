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
def add_variable_annotation(self, name: str, annotation: ast.AST) -> None:
    qualname = self.get_qualname_for(name)
    if qualname:
        basename = '.'.join(qualname[:-1])
        self.annotations[basename, name] = unparse(annotation)