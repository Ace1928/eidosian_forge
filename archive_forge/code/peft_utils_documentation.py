import importlib
import os
from typing import Dict, Optional, Union
from packaging import version
from .hub import cached_file
from .import_utils import is_peft_available

    Checks if the version of PEFT is compatible.

    Args:
        version (`str`):
            The version of PEFT to check against.
    