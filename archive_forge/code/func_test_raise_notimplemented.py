from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_raise_notimplemented(self):
    self.flakes('\n        raise NotImplementedError("This is fine")\n        ')
    self.flakes('\n        raise NotImplementedError\n        ')
    self.flakes('\n        raise NotImplemented("This isn\'t gonna work")\n        ', m.RaiseNotImplemented)
    self.flakes('\n        raise NotImplemented\n        ', m.RaiseNotImplemented)