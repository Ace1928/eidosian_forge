import os
import platform
import sys
import subprocess
import re
import warnings
from Bio import BiopythonDeprecationWarning
def _get_parameter(self, name):
    """Get a commandline option value (PRIVATE)."""
    for parameter in self.parameters:
        if name in parameter.names:
            if isinstance(parameter, _Switch):
                return parameter.is_set
            else:
                return parameter.value
    raise ValueError(f'Option name {name} was not found.')