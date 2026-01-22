from ..Node import Node
from .common import CtrlNode
class AbsNode(UniOpNode):
    """Returns abs(Inp). Does not check input types."""
    nodeName = 'Abs'

    def __init__(self, name):
        UniOpNode.__init__(self, name, '__abs__')