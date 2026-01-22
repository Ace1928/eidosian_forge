from reportlab.graphics.shapes import *
from reportlab.lib.validators import DerivedValue
from reportlab import rl_config
from . transform import mmult, inverse
def _expandUserNode(node, canvas):
    if isinstance(node, UserNode):
        try:
            if hasattr(node, '_canvas'):
                ocanvas = 1
            else:
                node._canvas = canvas
                ocanvas = None
            onode = node
            node = node.provideNode()
        finally:
            if not ocanvas:
                del onode._canvas
    return node