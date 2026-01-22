from __future__ import absolute_import
import cython
from io import StringIO
import re
import sys
from unicodedata import lookup as lookup_unicodechar, category as unicode_category
from functools import partial, reduce
from .Scanning import PyrexScanner, FileSourceDescriptor, tentatively_scan
from . import Nodes
from . import ExprNodes
from . import Builtin
from . import StringEncoding
from .StringEncoding import EncodedString, bytes_literal, _unicode, _bytes
from .ModuleNode import ModuleNode
from .Errors import error, warning
from .. import Utils
from . import Future
from . import Options
def _extract_docstring(node):
    """
    Extract a docstring from a statement or from the first statement
    in a list.  Remove the statement if found.  Return a tuple
    (plain-docstring or None, node).
    """
    doc_node = None
    if node is None:
        pass
    elif isinstance(node, Nodes.ExprStatNode):
        if node.expr.is_string_literal:
            doc_node = node.expr
            node = Nodes.StatListNode(node.pos, stats=[])
    elif isinstance(node, Nodes.StatListNode) and node.stats:
        stats = node.stats
        if isinstance(stats[0], Nodes.ExprStatNode):
            if stats[0].expr.is_string_literal:
                doc_node = stats[0].expr
                del stats[0]
    if doc_node is None:
        doc = None
    elif isinstance(doc_node, ExprNodes.BytesNode):
        warning(node.pos, 'Python 3 requires docstrings to be unicode strings')
        doc = doc_node.value
    elif isinstance(doc_node, ExprNodes.StringNode):
        doc = doc_node.unicode_value
        if doc is None:
            doc = doc_node.value
    else:
        doc = doc_node.value
    return (doc, node)