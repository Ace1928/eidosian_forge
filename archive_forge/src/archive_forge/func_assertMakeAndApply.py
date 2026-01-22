import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def assertMakeAndApply(self, source, target):
    """Assert that generating a delta and applying gives success."""
    delta = self.make_delta(source, target)
    bytes = self.apply_delta(source, delta)
    self.assertEqualDiff(target, bytes)