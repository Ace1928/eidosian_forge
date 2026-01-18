from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_used_package_with_submodule_import(self):
    """
        Usage of package marks submodule imports as used.
        """
    self.flakes('\n        import fu\n        import fu.bar\n        fu.x\n        ')
    self.flakes('\n        import fu.bar\n        import fu\n        fu.x\n        ')