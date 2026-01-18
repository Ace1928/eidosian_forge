import os
import platform
import subprocess
import sys
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, Union
from tqdm.auto import tqdm
from cmdstanpy import _DOT_CMDSTAN
from .. import progress as progbar
from .logging import get_logger
def validate_dir(install_dir: str) -> None:
    """Check that specified install directory exists, can write."""
    if not os.path.exists(install_dir):
        try:
            os.makedirs(install_dir)
        except (IOError, OSError, PermissionError) as e:
            raise ValueError('Cannot create directory: {}'.format(install_dir)) from e
    else:
        if not os.path.isdir(install_dir):
            raise ValueError('File exists, should be a directory: {}'.format(install_dir))
        try:
            with open('tmp_test_w', 'w'):
                pass
            os.remove('tmp_test_w')
        except OSError as e:
            raise ValueError('Cannot write files to directory {}'.format(install_dir)) from e