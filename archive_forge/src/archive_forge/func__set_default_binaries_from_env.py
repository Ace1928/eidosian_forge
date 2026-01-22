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
def _set_default_binaries_from_env(self) -> None:
    """Set default binaries from the environment.

        For example, pkg-config can be set via PKG_CONFIG, or in the machine
        file. We want to set the default to the env variable.
        """
    opts = itertools.chain(envconfig.DEPRECATED_ENV_PROG_MAP.items(), envconfig.ENV_VAR_PROG_MAP.items())
    for (name, evar), for_machine in itertools.product(opts, MachineChoice):
        p_env = _get_env_var(for_machine, self.is_cross_build(), evar)
        if p_env is not None:
            if os.path.exists(p_env):
                self.binaries[for_machine].binaries.setdefault(name, [p_env])
            else:
                self.binaries[for_machine].binaries.setdefault(name, mesonlib.split_args(p_env))