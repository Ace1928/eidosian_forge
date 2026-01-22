from typing import ClassVar, List, Sequence
from twisted.python.util import FancyEqMixin
class _ColorAttr(_Attribute):
    """
    Generic color attribute.

    @param color: Color value.

    @param ground: Foreground or background attribute name.
    """
    compareAttributes = ('color', 'ground', 'children')

    def __init__(self, color, ground):
        _Attribute.__init__(self)
        self.color = color
        self.ground = ground

    def serialize(self, write, attrs, attributeRenderer):
        attrs = attrs._withAttribute(self.ground, self.color)
        _Attribute.serialize(self, write, attrs, attributeRenderer)