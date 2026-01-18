from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_variable_annotation_references_self_name_undefined(self):
    self.flakes('\n        x: int = x\n        ', m.UndefinedName)