from io import StringIO
from twisted.python import text
from twisted.trial import unittest
class StringyStringTests(unittest.TestCase):
    """
    Tests for L{text.stringyString}.
    """

    def test_tuple(self) -> None:
        """
        Tuple elements are displayed on separate lines.
        """
        s = ('a', 'b')
        m = text.stringyString(s)
        self.assertEqual(m, '(a,\n b,)\n')

    def test_dict(self) -> None:
        """
        Dicts elements are displayed using C{str()}.
        """
        s = {'a': 0}
        m = text.stringyString(s)
        self.assertEqual(m, '{a: 0}')

    def test_list(self) -> None:
        """
        List elements are displayed on separate lines using C{str()}.
        """
        s = ['a', 'b']
        m = text.stringyString(s)
        self.assertEqual(m, '[a,\n b,]\n')