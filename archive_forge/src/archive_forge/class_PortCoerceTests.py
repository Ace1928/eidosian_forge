from twisted.python import usage
from twisted.trial import unittest
class PortCoerceTests(unittest.TestCase):
    """
    Test the behavior of L{usage.portCoerce}.
    """

    def test_validCoerce(self):
        """
        Test the answers with valid input.
        """
        self.assertEqual(0, usage.portCoerce('0'))
        self.assertEqual(3210, usage.portCoerce('3210'))
        self.assertEqual(65535, usage.portCoerce('65535'))

    def test_errorCoerce(self):
        """
        Test error path.
        """
        self.assertRaises(ValueError, usage.portCoerce, '')
        self.assertRaises(ValueError, usage.portCoerce, '-21')
        self.assertRaises(ValueError, usage.portCoerce, '212189')
        self.assertRaises(ValueError, usage.portCoerce, 'foo')