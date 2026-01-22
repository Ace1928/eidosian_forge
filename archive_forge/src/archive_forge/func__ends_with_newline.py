import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def _ends_with_newline(leaf, suffix=''):
    leaf = _skip_dedent_error_leaves(leaf)
    if leaf.type == 'error_leaf':
        typ = leaf.token_type.lower()
    else:
        typ = leaf.type
    return typ == 'newline' or suffix.endswith('\n') or suffix.endswith('\r')