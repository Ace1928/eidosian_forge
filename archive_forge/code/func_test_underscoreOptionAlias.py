from twisted.python import usage
from twisted.trial import unittest
def test_underscoreOptionAlias(self):
    """
        An option name with a dash in it can have an alias.
        """
    self.usage.parseOptions(['-u', 'bar'])
    self.assertEqual(self.usage.underscoreValue, 'bar')