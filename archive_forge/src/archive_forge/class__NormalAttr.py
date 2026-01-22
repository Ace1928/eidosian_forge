from typing import ClassVar, List, Sequence
from twisted.python.util import FancyEqMixin
class _NormalAttr(_Attribute):
    """
    A text attribute for normal text.
    """

    def serialize(self, write, attrs, attributeRenderer):
        attrs.__init__()
        _Attribute.serialize(self, write, attrs, attributeRenderer)