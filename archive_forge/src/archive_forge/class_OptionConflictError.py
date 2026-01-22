import sys, os
import textwrap
class OptionConflictError(OptionError):
    """
    Raised if conflicting options are added to an OptionParser.
    """