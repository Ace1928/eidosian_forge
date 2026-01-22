import unittest
from traits.api import (
from traits.observation.api import (
class DelegateMess(HasTraits):
    dummy1 = Instance(Dummy, args=())
    dummy2 = Instance(Dummy2)
    y = DelegatesTo('dummy2')
    handler_called = Bool(False)

    def _dummy2_default(self):
        return Dummy2(dummy=self.dummy1)

    @observe('dummy1.x')
    def _on_dummy1_x(self, event):
        self.handler_called = True