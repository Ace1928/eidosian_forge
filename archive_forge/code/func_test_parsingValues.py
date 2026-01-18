from twisted.python import usage
from twisted.trial import unittest
def test_parsingValues(self):
    """
        int and float values are parsed.
        """
    argV = '--fooint 912 --foofloat -823.1 --eggint 32 --eggfloat 21'.split()
    self.usage.parseOptions(argV)
    self.assertEqual(self.usage.opts['fooint'], 912)
    self.assertIsInstance(self.usage.opts['fooint'], int)
    self.assertEqual(self.usage.opts['foofloat'], -823.1)
    self.assertIsInstance(self.usage.opts['foofloat'], float)
    self.assertEqual(self.usage.opts['eggint'], 32)
    self.assertIsInstance(self.usage.opts['eggint'], int)
    self.assertEqual(self.usage.opts['eggfloat'], 21.0)
    self.assertIsInstance(self.usage.opts['eggfloat'], float)