from typing import ClassVar, List, Sequence
from twisted.python.util import FancyEqMixin
class _BackgroundColorAttr(_ColorAttr):
    """
    Background color attribute.
    """

    def __init__(self, color):
        _ColorAttr.__init__(self, color, 'background')