import ctypes as ct
import errno
import os
from pathlib import Path
import platform
from typing import Set, Union
from warnings import warn
import torch
from .env_vars import get_potentially_lib_path_containing_env_vars
def add_log_entry(self, msg, is_warning=False):
    self.cuda_setup_log.append((msg, is_warning))