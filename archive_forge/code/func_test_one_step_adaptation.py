import unittest
from traits.adaptation.api import AdaptationManager
import traits.adaptation.tests.abc_examples
import traits.adaptation.tests.interface_examples
def test_one_step_adaptation(self):
    ex = self.examples
    self.adaptation_manager.register_factory(factory=ex.UKStandardToEUStandard, from_protocol=ex.UKStandard, to_protocol=ex.EUStandard)
    plug = ex.UKPlug()
    eu_plug = self.adaptation_manager.adapt(plug, ex.EUStandard)
    self.assertIsNotNone(eu_plug)
    self.assertIsInstance(eu_plug, ex.UKStandardToEUStandard)
    eu_plug = self.adaptation_manager.adapt(plug, ex.EUPlug, None)
    self.assertIsNone(eu_plug)