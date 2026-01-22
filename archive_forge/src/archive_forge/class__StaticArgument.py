import os
import platform
import sys
import subprocess
import re
import warnings
from Bio import BiopythonDeprecationWarning
class _StaticArgument(_AbstractParameter):
    """Represent a static (read only) argument on a commandline.

    This is not intended to be exposed as a named argument or
    property of a command line wrapper object.
    """

    def __init__(self, value):
        self.names = []
        self.is_required = False
        self.is_set = True
        self.value = value

    def __str__(self):
        return f'{self.value} '