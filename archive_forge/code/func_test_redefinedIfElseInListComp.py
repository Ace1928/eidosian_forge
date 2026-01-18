from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_redefinedIfElseInListComp(self):
    """
        Test that shadowing a variable in a list comprehension in
        an if and else block does not raise a warning.
        """
    self.flakes("\n        if False:\n            a = 1\n        else:\n            [a for a in '12']\n        ")