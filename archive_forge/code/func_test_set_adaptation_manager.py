import unittest
from traits.adaptation.api import (
import traits.adaptation.tests.abc_examples
def test_set_adaptation_manager(self):
    ex = self.examples
    adaptation_manager = AdaptationManager()
    adaptation_manager.register_factory(factory=ex.UKStandardToEUStandard, from_protocol=ex.UKStandard, to_protocol=ex.EUStandard)
    uk_plug = ex.UKPlug()
    set_global_adaptation_manager(adaptation_manager)
    global_adaptation_manager = get_global_adaptation_manager()
    eu_plug = global_adaptation_manager.adapt(uk_plug, ex.EUStandard)
    self.assertIsNotNone(eu_plug)
    self.assertIsInstance(eu_plug, ex.UKStandardToEUStandard)