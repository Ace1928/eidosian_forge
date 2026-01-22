import sys, os
import textwrap
class BadOptionError(OptParseError):
    """
    Raised if an invalid option is seen on the command line.
    """

    def __init__(self, opt_str):
        self.opt_str = opt_str

    def __str__(self):
        return _('no such option: %s') % self.opt_str