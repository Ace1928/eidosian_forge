from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_redefinedTryNested(self):
    """
        Test that importing a module twice using a nested
        try/except and if blocks does not issue a warning.
        """
    self.flakes('\n        try:\n            if True:\n                if True:\n                    import os\n        except:\n            import os\n        os.path')