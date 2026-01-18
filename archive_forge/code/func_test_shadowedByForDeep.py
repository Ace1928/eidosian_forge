from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_shadowedByForDeep(self):
    """
        Test that shadowing a global name with a for loop variable nested in a
        tuple unpack generates a warning.
        """
    self.flakes('\n        import fu\n        fu.bar()\n        for (x, y, z, (a, b, c, (fu,))) in ():\n            pass\n        ', m.ImportShadowedByLoopVar)
    self.flakes('\n        import fu\n        fu.bar()\n        for [x, y, z, (a, b, c, (fu,))] in ():\n            pass\n        ', m.ImportShadowedByLoopVar)