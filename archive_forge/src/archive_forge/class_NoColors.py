import os
import warnings
from IPython.utils.ipstruct import Struct
class NoColors:
    """This defines all the same names as the colour classes, but maps them to
    empty strings, so it can easily be substituted to turn off colours."""
    NoColor = ''
    Normal = ''