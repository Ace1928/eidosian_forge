from twisted.python import usage
from twisted.trial import unittest
class OptionsInternalTests(unittest.TestCase):
    """
    Tests internal behavior of C{usage.Options}.
    """

    def test_optionsAliasesOrder(self):
        """
        Options which are synonyms to another option are aliases towards the
        longest option name.
        """

        class Opts(usage.Options):

            def opt_very_very_long(self):
                """
                This is an option method with a very long name, that is going to
                be aliased.
                """
            opt_short = opt_very_very_long
            opt_s = opt_very_very_long
        opts = Opts()
        self.assertEqual(dict.fromkeys(['s', 'short', 'very-very-long'], 'very-very-long'), {'s': opts.synonyms['s'], 'short': opts.synonyms['short'], 'very-very-long': opts.synonyms['very-very-long']})