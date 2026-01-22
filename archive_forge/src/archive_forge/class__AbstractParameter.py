import os
import platform
import sys
import subprocess
import re
import warnings
from Bio import BiopythonDeprecationWarning
class _AbstractParameter:
    """A class to hold information about a parameter for a commandline.

    Do not use this directly, instead use one of the subclasses.
    """

    def __init__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError