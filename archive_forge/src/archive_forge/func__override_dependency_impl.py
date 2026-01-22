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
def _override_dependency_impl(self, name: str, dep: dependencies.Dependency, kwargs: 'FuncOverrideDependency', static: T.Optional[bool], permissive: bool=False) -> None:
    nkwargs = T.cast('T.Dict[str, T.Any]', kwargs.copy())
    if static is None:
        del nkwargs['static']
    else:
        nkwargs['static'] = static
    identifier = dependencies.get_dep_identifier(name, nkwargs)
    for_machine = kwargs['native']
    override = self.build.dependency_overrides[for_machine].get(identifier)
    if override:
        if permissive:
            return
        m = 'Tried to override dependency {!r} which has already been resolved or overridden at {}'
        location = mlog.get_error_location_string(override.node.filename, override.node.lineno)
        raise InterpreterException(m.format(name, location))
    self.build.dependency_overrides[for_machine][identifier] = build.DependencyOverride(dep, self.interpreter.current_node)