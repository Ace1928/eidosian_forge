import sys, os
import textwrap
class OptionValueError(OptParseError):
    """
    Raised if an invalid option value is encountered on the command
    line.
    """