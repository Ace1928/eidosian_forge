from __future__ import annotations
import os
import typing as T
from .. import mesonlib
from .. import dependencies
from .. import build
from .. import mlog, coredata
from ..mesonlib import MachineChoice, OptionKey
from ..programs import OverrideProgram, ExternalProgram
from ..interpreter.type_checking import ENV_KW, ENV_METHOD_KW, ENV_SEPARATOR_KW, env_convertor_with_method
from ..interpreterbase import (MesonInterpreterObject, FeatureNew, FeatureDeprecated,
from .primitives import MesonVersionString
from .type_checking import NATIVE_KW, NoneType
def _process_script_args(self, name: str, args: T.Sequence[T.Union[str, mesonlib.File, build.BuildTarget, build.CustomTarget, build.CustomTargetIndex, ExternalProgram]]) -> T.List[str]:
    script_args = []
    new = False
    for a in args:
        if isinstance(a, str):
            script_args.append(a)
        elif isinstance(a, mesonlib.File):
            new = True
            script_args.append(a.rel_to_builddir(self.interpreter.environment.source_dir))
        elif isinstance(a, (build.BuildTarget, build.CustomTarget, build.CustomTargetIndex)):
            new = True
            script_args.extend([os.path.join(a.get_subdir(), o) for o in a.get_outputs()])
            if isinstance(a, build.CustomTargetIndex):
                a.target.build_by_default = True
            else:
                a.build_by_default = True
        else:
            script_args.extend(a.command)
            new = True
    if new:
        FeatureNew.single_use(f'Calling "{name}" with File, CustomTarget, Index of CustomTarget, Executable, or ExternalProgram', '0.55.0', self.interpreter.subproject, location=self.current_node)
    return script_args