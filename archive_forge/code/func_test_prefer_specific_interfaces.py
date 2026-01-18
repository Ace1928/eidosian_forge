import unittest
from traits.adaptation.api import AdaptationManager
import traits.adaptation.tests.abc_examples
import traits.adaptation.tests.interface_examples
def test_prefer_specific_interfaces(self):
    ex = self.examples
    self.adaptation_manager.register_factory(factory=ex.IIntermediateToITarget, from_protocol=ex.IIntermediate, to_protocol=ex.ITarget)
    self.adaptation_manager.register_factory(factory=ex.IHumanToIIntermediate, from_protocol=ex.IHuman, to_protocol=ex.IIntermediate)
    self.adaptation_manager.register_factory(factory=ex.IChildToIIntermediate, from_protocol=ex.IChild, to_protocol=ex.IIntermediate)
    self.adaptation_manager.register_factory(factory=ex.IPrimateToIIntermediate, from_protocol=ex.IPrimate, to_protocol=ex.IIntermediate)
    source = ex.Source()
    target = self.adaptation_manager.adapt(source, ex.ITarget)
    self.assertIsNotNone(target)
    self.assertIs(type(target.adaptee), ex.IChildToIIntermediate)