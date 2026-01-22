from ..Node import Node
from .common import CtrlNode
class SubtractNode(BinOpNode):
    """Returns A - B. Does not check input types."""
    nodeName = 'Subtract'

    def __init__(self, name):
        BinOpNode.__init__(self, name, '__sub__')