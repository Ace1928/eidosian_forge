import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def _assert_nodes_are_equal(node1, node2):
    try:
        children1 = node1.children
    except AttributeError:
        assert not hasattr(node2, 'children'), (node1, node2)
        assert node1.value == node2.value, (node1, node2)
        assert node1.type == node2.type, (node1, node2)
        assert node1.prefix == node2.prefix, (node1, node2)
        assert node1.start_pos == node2.start_pos, (node1, node2)
        return
    else:
        try:
            children2 = node2.children
        except AttributeError:
            assert False, (node1, node2)
    for n1, n2 in zip(children1, children2):
        _assert_nodes_are_equal(n1, n2)
    assert len(children1) == len(children2), '\n' + repr(children1) + '\n' + repr(children2)