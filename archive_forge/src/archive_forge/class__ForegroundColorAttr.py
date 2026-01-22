from typing import ClassVar, List, Sequence
from twisted.python.util import FancyEqMixin
class _ForegroundColorAttr(_ColorAttr):
    """
    Foreground color attribute.
    """

    def __init__(self, color):
        _ColorAttr.__init__(self, color, 'foreground')