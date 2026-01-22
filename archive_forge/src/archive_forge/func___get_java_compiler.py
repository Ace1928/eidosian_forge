from __future__ import annotations
import pathlib
import typing as T
from mesonbuild import mesonlib
from mesonbuild.build import CustomTarget, CustomTargetIndex, GeneratedList, Target
from mesonbuild.compilers import detect_compiler_for
from mesonbuild.interpreterbase.decorators import ContainerTypeInfo, FeatureDeprecated, FeatureNew, KwargInfo, typed_pos_args, typed_kwargs
from mesonbuild.mesonlib import version_compare, MachineChoice
from . import NewExtensionModule, ModuleReturnValue, ModuleInfo
from ..interpreter.type_checking import NoneType
def __get_java_compiler(self, state: ModuleState) -> Compiler:
    if 'java' not in state.environment.coredata.compilers[MachineChoice.BUILD]:
        detect_compiler_for(state.environment, 'java', MachineChoice.BUILD, False, state.subproject)
    return state.environment.coredata.compilers[MachineChoice.BUILD]['java']