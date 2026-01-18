import re
from collections import namedtuple
from textwrap import dedent
from itertools import chain
from functools import wraps
from inspect import Parameter
from parso.python.parser import Parser
from parso.python import tree
from jedi.inference.base_value import NO_VALUES
from jedi.inference.syntax_tree import infer_atom
from jedi.inference.helpers import infer_call_of_leaf
from jedi.inference.compiled import get_string_value_set
from jedi.cache import signature_time_cache, memoize_method
from jedi.parser_utils import get_parent_scope
def tokenize_without_endmarker(code):
    tokens = grammar._tokenize(code)
    for token in tokens:
        if token.string == safeword:
            raise EndMarkerReached()
        elif token.prefix.endswith(safeword):
            raise EndMarkerReached()
        elif token.string.endswith(safeword):
            yield token
            raise EndMarkerReached()
        else:
            yield token