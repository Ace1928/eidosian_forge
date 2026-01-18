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
def validate_cmdstan_path(path: str) -> None:
    """
    Validate that CmdStan directory exists and binaries have been built.
    Throws exception if specified path is invalid.
    """
    if not os.path.isdir(path):
        raise ValueError(f'No CmdStan directory, path {path} does not exist.')
    if not os.path.exists(os.path.join(path, 'bin', 'stanc' + EXTENSION)):
        raise ValueError(f'CmdStan installataion missing binaries in {path}/bin. Re-install cmdstan by running command "install_cmdstan --overwrite", or Python code "import cmdstanpy; cmdstanpy.install_cmdstan(overwrite=True)"')