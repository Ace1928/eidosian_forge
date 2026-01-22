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
def add_definition(self, name: str, entry: Tuple[str, int, int]) -> None:
    """Add a location of definition."""
    if self.indents and self.indents[-1][0] == 'def' and (entry[0] == 'def'):
        pass
    else:
        self.definitions[name] = entry