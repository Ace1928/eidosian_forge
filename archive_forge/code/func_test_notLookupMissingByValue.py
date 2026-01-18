import warnings
from twisted.trial.unittest import TestCase
def test_notLookupMissingByValue(self):
    """
        L{Flags.lookupByValue} raises L{ValueError} when called with a value
        with which no constant is associated.
        """
    self.assertRaises(ValueError, self.FXF.lookupByValue, 16)