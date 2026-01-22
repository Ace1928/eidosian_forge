import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def _gp_float(tok):
    """Get a float from a token, if it fails, returns the string (PRIVATE)."""
    try:
        return float(tok)
    except ValueError:
        return str(tok)