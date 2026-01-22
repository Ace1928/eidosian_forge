from __future__ import annotations
from pathlib import Path
import os
import shlex
import subprocess
import typing as T
from . import ExtensionModule, ModuleReturnValue, NewExtensionModule, ModuleInfo
from .. import mlog, build
from ..compilers.compilers import CFLAGS_MAPPING
from ..envconfig import ENV_VAR_PROG_MAP
from ..dependencies import InternalDependency
from ..dependencies.pkgconfig import PkgConfigInterface
from ..interpreterbase import FeatureNew
from ..interpreter.type_checking import ENV_KW, DEPENDS_KW
from ..interpreterbase.decorators import ContainerTypeInfo, KwargInfo, typed_kwargs, typed_pos_args
from ..mesonlib import (EnvironmentException, MesonException, Popen_safe, MachineChoice,
def _validate_configure_options(self, variables: T.List[T.Tuple[str, str, str]], state: 'ModuleState') -> None:
    for key, default, val in variables:
        if default is None:
            continue
        key_format = f'@{key}@'
        for option in self.configure_options:
            if key_format in option:
                break
        else:
            FeatureNew('Default configure_option', '0.57.0').use(self.subproject, state.current_node)
            self.configure_options.append(default)