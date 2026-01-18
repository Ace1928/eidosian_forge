import io
import json
import os
import platform
import shutil
import subprocess
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
from cmdstanpy.utils import get_logger
from cmdstanpy.utils.cmdstan import (
from cmdstanpy.utils.command import do_command
from cmdstanpy.utils.filesystem import SanitizedOrTmpFilePath
def validate_cpp_opts(self) -> None:
    """
        Check cpp compiler args.
        Raise ValueError if bad config is found.
        """
    if self._cpp_options is None:
        return
    for key in ['OPENCL_DEVICE_ID', 'OPENCL_PLATFORM_ID']:
        if key in self._cpp_options:
            self._cpp_options['STAN_OPENCL'] = 'TRUE'
            val = self._cpp_options[key]
            if not isinstance(val, int) or val < 0:
                raise ValueError(f'{key} must be a non-negative integer value, found {val}.')