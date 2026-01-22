from typing import ClassVar, List, Sequence
from twisted.python.util import FancyEqMixin
class DefaultFormattingState(FancyEqMixin):
    """
    A character attribute that does nothing, thus applying no attributes to
    text.
    """
    compareAttributes: ClassVar[Sequence[str]] = ('_dummy',)
    _dummy = 0

    def copy(self):
        """
        Make a copy of this formatting state.

        @return: A formatting state instance.
        """
        return type(self)()

    def _withAttribute(self, name, value):
        """
        Add a character attribute to a copy of this formatting state.

        @param name: Attribute name to be added to formatting state.

        @param value: Attribute value.

        @return: A formatting state instance with the new attribute.
        """
        return self.copy()

    def toVT102(self):
        """
        Emit a VT102 control sequence that will set up all the attributes this
        formatting state has set.

        @return: A string containing VT102 control sequences that mimic this
            formatting state.
        """
        return ''