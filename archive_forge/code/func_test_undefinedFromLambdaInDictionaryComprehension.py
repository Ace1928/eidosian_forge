import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_undefinedFromLambdaInDictionaryComprehension(self):
    """
        Undefined name referenced from a lambda function within a dict/set
        comprehension.
        """
    self.flakes('\n        {lambda: id(y) for x in range(10)}\n        ', m.UndefinedName)