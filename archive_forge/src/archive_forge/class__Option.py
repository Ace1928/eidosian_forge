import os
import platform
import sys
import subprocess
import re
import warnings
from Bio import BiopythonDeprecationWarning
class _Option(_AbstractParameter):
    """Represent an option that can be set for a program.

    This holds UNIXish options like --append=yes and -a yes,
    where a value (here "yes") is generally expected.

    For UNIXish options like -kimura in clustalw which don't
    take a value, use the _Switch object instead.

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
     - filename -- True if this argument is a filename (or other argument
       that should be quoted) and should be automatically quoted if it
       contains spaces.
     - checker_function -- a reference to a function that will determine
       if a given value is valid for this parameter. This function can either
       raise an error when given a bad value, or return a [0, 1] decision on
       whether the value is correct.
     - equate -- should an equals sign be inserted if a value is used?
     - is_required -- a flag to indicate if the parameter must be set for
       the program to be run.
     - is_set -- if the parameter has been set
     - value -- the value of a parameter

    """

    def __init__(self, names, description, filename=False, checker_function=None, is_required=False, equate=True):
        self.names = names
        if not isinstance(description, str):
            raise TypeError(f'Should be a string: {description!r} for {names[-1]}')
        self.is_filename = filename
        self.checker_function = checker_function
        self.description = description
        self.equate = equate
        self.is_required = is_required
        self.is_set = False
        self.value = None

    def __str__(self):
        """Return the value of this option for the commandline.

        Includes a trailing space.
        """
        if self.value is None:
            return f'{self.names[0]} '
        if self.is_filename:
            v = _escape_filename(self.value)
        else:
            v = str(self.value)
        if self.equate:
            return f'{self.names[0]}={v} '
        else:
            return f'{self.names[0]} {v} '