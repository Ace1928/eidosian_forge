import os
import warnings
from IPython.utils.ipstruct import Struct
def add_scheme(self, new_scheme):
    """Add a new color scheme to the table."""
    if not isinstance(new_scheme, ColorScheme):
        raise ValueError('ColorSchemeTable only accepts ColorScheme instances')
    self[new_scheme.name] = new_scheme