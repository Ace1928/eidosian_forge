import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def _assert_valid_graph(node):
    """
    Checks if the parent/children relationship is correct.

    This is a check that only runs during debugging/testing.
    """
    try:
        children = node.children
    except AttributeError:
        if node.type == 'error_leaf' and node.token_type in _INDENTATION_TOKENS:
            assert not node.value
            assert not node.prefix
            return
        previous_leaf = _get_previous_leaf_if_indentation(node.get_previous_leaf())
        if previous_leaf is None:
            content = node.prefix
            previous_start_pos = (1, 0)
        else:
            assert previous_leaf.end_pos <= node.start_pos, (previous_leaf, node)
            content = previous_leaf.value + node.prefix
            previous_start_pos = previous_leaf.start_pos
        if '\n' in content or '\r' in content:
            splitted = split_lines(content)
            line = previous_start_pos[0] + len(splitted) - 1
            actual = (line, len(splitted[-1]))
        else:
            actual = (previous_start_pos[0], previous_start_pos[1] + len(content))
            if content.startswith(BOM_UTF8_STRING) and node.get_start_pos_of_prefix() == (1, 0):
                actual = (actual[0], actual[1] - 1)
        assert node.start_pos == actual, (node.start_pos, actual)
    else:
        for child in children:
            assert child.parent == node, (node, child)
            _assert_valid_graph(child)