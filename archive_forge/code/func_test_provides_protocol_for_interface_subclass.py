import unittest
from traits.adaptation.api import AdaptationManager
import traits.adaptation.tests.abc_examples
import traits.adaptation.tests.interface_examples
def test_provides_protocol_for_interface_subclass(self):
    from traits.api import Interface

    class IA(Interface):
        pass

    class IB(IA):
        pass
    self.assertTrue(self.adaptation_manager.provides_protocol(IB, IA))