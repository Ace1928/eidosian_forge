import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_undefinedWithErrorHandler(self):
    """
        Some compatibility code checks explicitly for NameError.
        It should not trigger warnings.
        """
    self.flakes('\n        try:\n            socket_map\n        except NameError:\n            socket_map = {}\n        ')
    self.flakes('\n        try:\n            _memoryview.contiguous\n        except (NameError, AttributeError):\n            raise RuntimeError("Python >= 3.3 is required")\n        ')
    self.flakes('\n        try:\n            socket_map\n        except:\n            socket_map = {}\n        ', m.UndefinedName)
    self.flakes('\n        try:\n            socket_map\n        except Exception:\n            socket_map = {}\n        ', m.UndefinedName)