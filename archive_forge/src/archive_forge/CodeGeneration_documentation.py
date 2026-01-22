from __future__ import absolute_import
from .Visitor import VisitorTransform
from .Nodes import StatListNode

    Finds nodes in a pxd file that should generate code, and
    returns them in a StatListNode.

    The result is a tuple (StatListNode, ModuleScope), i.e.
    everything that is needed from the pxd after it is processed.

    A purer approach would be to separately compile the pxd code,
    but the result would have to be slightly more sophisticated
    than pure strings (functions + wanted interned strings +
    wanted utility code + wanted cached objects) so for now this
    approach is taken.
    