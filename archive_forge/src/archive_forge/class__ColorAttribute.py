from typing import ClassVar, List, Sequence
from twisted.python.util import FancyEqMixin
class _ColorAttribute:
    """
    A color text attribute.

    Attribute access results in a color value lookup, by name, in
    I{_ColorAttribute.attrs}.

    @type ground: L{_ColorAttr}
    @param ground: Foreground or background color attribute to look color names
        up from.

    @param attrs: Mapping of color names to color values.
    @type attrs: Dict like object.
    """

    def __init__(self, ground, attrs):
        self.ground = ground
        self.attrs = attrs

    def __getattr__(self, name):
        try:
            return self.ground(self.attrs[name])
        except KeyError:
            raise AttributeError(name)