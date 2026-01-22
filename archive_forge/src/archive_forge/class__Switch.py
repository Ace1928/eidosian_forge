import os
import platform
import sys
import subprocess
import re
import warnings
from Bio import BiopythonDeprecationWarning
class _Switch(_AbstractParameter):
    """Represent an optional argument switch for a program.

    This holds UNIXish options like -kimura in clustalw which don't
    take a value, they are either included in the command string
    or omitted.

    Attributes:
     - names -- a list of string names (typically two entries) by which
       the parameter can be set via the legacy set_parameter method
       (eg ["-a", "--append", "append"]). The first name in list is used
       when building the command line. The last name in the list is a
       "human readable" name describing the option in one word. This
       must be a valid Python identifier as it is used as the property
       name and as a keyword argument, and should therefore follow PEP8
       naming.
     - description -- a description of the option. This is used as
       the property docstring.
     - is_set -- if the parameter has been set

    NOTE - There is no value attribute, see is_set instead,

    """

    def __init__(self, names, description):
        self.names = names
        self.description = description
        self.is_set = False
        self.is_required = False

    def __str__(self):
        """Return the value of this option for the commandline.

        Includes a trailing space.
        """
        assert not hasattr(self, 'value')
        if self.is_set:
            return f'{self.names[0]} '
        else:
            return ''