import os
import shutil
import sys
from pathlib import Path
from typing import Union
from pyproj._datadir import (  # noqa: F401  pylint: disable=unused-import
from pyproj.exceptions import DataDirError
def valid_data_dir(potential_data_dir):
    if potential_data_dir is not None and Path(potential_data_dir, 'proj.db').exists():
        return True
    return False