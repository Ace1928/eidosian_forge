from twisted.python import usage
from twisted.trial import unittest
class ParseCorrectnessTests(unittest.TestCase):
    """
    Test L{usage.Options.parseOptions} for correct values under
    good conditions.
    """

    def setUp(self):
        """
        Instantiate and parseOptions a well-behaved Options class.
        """
        self.niceArgV = '--long Alpha -n Beta --shortless Gamma -f --myflag --myparam Tofu'.split()
        self.nice = WellBehaved()
        self.nice.parseOptions(self.niceArgV)

    def test_checkParameters(self):
        """
        Parameters have correct values.
        """
        self.assertEqual(self.nice.opts['long'], 'Alpha')
        self.assertEqual(self.nice.opts['another'], 'Beta')
        self.assertEqual(self.nice.opts['longonly'], 'noshort')
        self.assertEqual(self.nice.opts['shortless'], 'Gamma')

    def test_checkFlags(self):
        """
        Flags have correct values.
        """
        self.assertEqual(self.nice.opts['aflag'], 1)
        self.assertEqual(self.nice.opts['flout'], 0)

    def test_checkCustoms(self):
        """
        Custom flags and parameters have correct values.
        """
        self.assertEqual(self.nice.opts['myflag'], 'PONY!')
        self.assertEqual(self.nice.opts['myparam'], 'Tofu WITH A PONY!')