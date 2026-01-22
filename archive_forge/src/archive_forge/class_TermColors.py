import os
import warnings
from IPython.utils.ipstruct import Struct
class TermColors:
    """Color escape sequences.

    This class defines the escape sequences for all the standard (ANSI?)
    colors in terminals. Also defines a NoColor escape which is just the null
    string, suitable for defining 'dummy' color schemes in terminals which get
    confused by color escapes.

    This class should be used as a mixin for building color schemes."""
    NoColor = ''
    Normal = '\x1b[0m'
    _base = '\x1b[%sm'