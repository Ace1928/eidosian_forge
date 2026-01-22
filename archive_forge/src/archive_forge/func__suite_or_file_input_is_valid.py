import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def _suite_or_file_input_is_valid(pgen_grammar, stack):
    if not _flows_finished(pgen_grammar, stack):
        return False
    for stack_node in reversed(stack):
        if stack_node.nonterminal == 'decorator':
            return False
        if stack_node.nonterminal == 'suite':
            return len(stack_node.nodes) > 1
    return True