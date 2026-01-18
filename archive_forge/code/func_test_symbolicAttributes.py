import warnings
from twisted.trial.unittest import TestCase
def test_symbolicAttributes(self):
    """
        Each name associated with a L{FlagConstant} instance in the definition
        of a L{Flags} subclass is available as an attribute on the resulting
        class.
        """
    self.assertTrue(hasattr(self.FXF, 'READ'))
    self.assertTrue(hasattr(self.FXF, 'WRITE'))
    self.assertTrue(hasattr(self.FXF, 'APPEND'))
    self.assertTrue(hasattr(self.FXF, 'EXCLUSIVE'))
    self.assertTrue(hasattr(self.FXF, 'TEXT'))