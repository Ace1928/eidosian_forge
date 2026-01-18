import warnings
from twisted.trial.unittest import TestCase
def test_truthiness(self):
    """
        Empty flags is false, non-empty flags is true.
        """
    self.assertTrue(self.FXF.WRITE)
    self.assertTrue(self.FXF.WRITE | self.FXF.EXCLUSIVE)
    self.assertFalse(self.FXF.WRITE & self.FXF.EXCLUSIVE)