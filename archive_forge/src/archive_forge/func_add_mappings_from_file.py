import os as _os
import sys as _sys
import warnings as _warnings
from .base import Sign
from .controller_db import mapping_list
def add_mappings_from_file(filename) -> None:
    """Add mappings from a file.

    Given a file path, open and parse the file for mappings.

    :Parameters:
        `filename` : str
            A file path.
    """
    with open(filename) as f:
        add_mappings_from_string(f.read())