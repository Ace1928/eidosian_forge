import logging
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from shutil import which
from typing import List, Optional
import torch
from packaging.version import parse
def _nvidia_smi():
    """
    Returns the right nvidia-smi command based on the system.
    """
    if platform.system() == 'Windows':
        command = which('nvidia-smi')
        if command is None:
            command = '%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe' % os.environ['systemdrive']
    else:
        command = 'nvidia-smi'
    return command