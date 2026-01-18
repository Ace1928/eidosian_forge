from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def tree_to_treesegment(canvas, t, make_node=TextWidget, make_leaf=TextWidget, **attribs):
    """
    Convert a Tree into a ``TreeSegmentWidget``.

    :param make_node: A ``CanvasWidget`` constructor or a function that
        creates ``CanvasWidgets``.  ``make_node`` is used to convert
        the Tree's nodes into ``CanvasWidgets``.  If no constructor
        is specified, then ``TextWidget`` will be used.
    :param make_leaf: A ``CanvasWidget`` constructor or a function that
        creates ``CanvasWidgets``.  ``make_leaf`` is used to convert
        the Tree's leafs into ``CanvasWidgets``.  If no constructor
        is specified, then ``TextWidget`` will be used.
    :param attribs: Attributes for the canvas widgets that make up the
        returned ``TreeSegmentWidget``.  Any attribute beginning with
        ``'tree_'`` will be passed to all ``TreeSegmentWidgets`` (with
        the ``'tree_'`` prefix removed.  Any attribute beginning with
        ``'node_'`` will be passed to all nodes.  Any attribute
        beginning with ``'leaf_'`` will be passed to all leaves.  And
        any attribute beginning with ``'loc_'`` will be passed to all
        text locations (for Trees).
    """
    tree_attribs = {}
    node_attribs = {}
    leaf_attribs = {}
    loc_attribs = {}
    for key, value in list(attribs.items()):
        if key[:5] == 'tree_':
            tree_attribs[key[5:]] = value
        elif key[:5] == 'node_':
            node_attribs[key[5:]] = value
        elif key[:5] == 'leaf_':
            leaf_attribs[key[5:]] = value
        elif key[:4] == 'loc_':
            loc_attribs[key[4:]] = value
        else:
            raise ValueError('Bad attribute: %s' % key)
    return _tree_to_treeseg(canvas, t, make_node, make_leaf, tree_attribs, node_attribs, leaf_attribs, loc_attribs)