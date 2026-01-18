from twisted.python import usage
from twisted.trial import unittest
def test_validCoerce(self):
    """
        Test the answers with valid input.
        """
    self.assertEqual(0, usage.portCoerce('0'))
    self.assertEqual(3210, usage.portCoerce('3210'))
    self.assertEqual(65535, usage.portCoerce('65535'))