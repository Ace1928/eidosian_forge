from __future__ import annotations
import itertools
import os, platform, re, sys, shutil
import typing as T
import collections
from . import coredata
from . import mesonlib
from .mesonlib import (
from . import mlog
from .programs import ExternalProgram
from .envconfig import (
from . import compilers
from .compilers import (
from functools import lru_cache
from mesonbuild import envconfig
def _get_env_var(for_machine: MachineChoice, is_cross: bool, var_name: str) -> T.Optional[str]:
    """
    Returns the exact env var and the value.
    """
    candidates = PerMachine([var_name + '_FOR_BUILD'] if is_cross else [var_name], [var_name])[for_machine]
    for var in candidates:
        value = os.environ.get(var)
        if value is not None:
            break
    else:
        formatted = ', '.join([f'{var!r}' for var in candidates])
        mlog.debug(f'None of {formatted} are defined in the environment, not changing global flags.')
        return None
    mlog.debug(f'Using {var!r} from environment with value: {value!r}')
    return value