import unittest
from traits.adaptation.api import (
import traits.adaptation.tests.abc_examples
def test_reset_adaptation_manager(self):
    ex = self.examples
    adaptation_manager = get_global_adaptation_manager()
    adaptation_manager.register_factory(factory=ex.UKStandardToEUStandard, from_protocol=ex.UKStandard, to_protocol=ex.EUStandard)
    uk_plug = ex.UKPlug()
    reset_global_adaptation_manager()
    adaptation_manager = get_global_adaptation_manager()
    with self.assertRaises(AdaptationError):
        adaptation_manager.adapt(uk_plug, ex.EUStandard)