import warnings
from twisted.trial.unittest import TestCase
def test_withoutOtherAttributes(self):
    """
        As usual, names not defined in the class scope of a L{Flags} subclass
        are not available as attributes on the resulting class.
        """
    self.assertFalse(hasattr(self.FXF, 'foo'))