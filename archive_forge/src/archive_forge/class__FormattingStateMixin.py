from typing import ClassVar, List, Sequence
from twisted.python.util import FancyEqMixin
class _FormattingStateMixin(DefaultFormattingState):
    """
    Mixin for the formatting state/attributes of a single character.
    """

    def copy(self):
        c = DefaultFormattingState.copy(self)
        c.__dict__.update(vars(self))
        return c

    def _withAttribute(self, name, value):
        if getattr(self, name) != value:
            attr = self.copy()
            attr._subtracting = not value
            setattr(attr, name, value)
            return attr
        else:
            return self.copy()