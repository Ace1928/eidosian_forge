import ctypes as ct
import errno
import os
from pathlib import Path
import platform
from typing import Set, Union
from warnings import warn
import torch
from .env_vars import get_potentially_lib_path_containing_env_vars

        Searches for a cuda installations, in the following order of priority:
            1. active conda env
            2. LD_LIBRARY_PATH
            3. any other env vars, while ignoring those that
                - are known to be unrelated (see `bnb.cuda_setup.env_vars.to_be_ignored`)
                - don't contain the path separator `/`

        If multiple libraries are found in part 3, we optimistically try one,
        while giving a warning message.
    