import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def _get_previous_leaf_if_indentation(leaf):
    while leaf and _is_indentation_error_leaf(leaf):
        leaf = leaf.get_previous_leaf()
    return leaf