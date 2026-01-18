import os
import shutil
import sys
from pathlib import Path
from typing import Union
from pyproj._datadir import (  # noqa: F401  pylint: disable=unused-import
from pyproj.exceptions import DataDirError
def valid_data_dirs(potential_data_dirs):
    if potential_data_dirs is None:
        return False
    for proj_data_dir in potential_data_dirs.split(os.pathsep):
        if valid_data_dir(proj_data_dir):
            return True
    return None