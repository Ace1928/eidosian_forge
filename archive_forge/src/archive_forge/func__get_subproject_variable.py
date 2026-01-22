from __future__ import annotations
from .interpreterobjects import extract_required_kwarg
from .. import mlog
from .. import dependencies
from .. import build
from ..wrap import WrapMode
from ..mesonlib import OptionKey, extract_as_list, stringlistify, version_compare_many, listify
from ..dependencies import Dependency, DependencyException, NotFoundDependency
from ..interpreterbase import (MesonInterpreterObject, FeatureNew,
import typing as T
def _get_subproject_variable(self, subproject: SubprojectHolder, varname: str) -> T.Optional[Dependency]:
    try:
        var_dep = subproject.get_variable_method([varname], {})
    except InvalidArguments:
        var_dep = None
    if not isinstance(var_dep, Dependency):
        mlog.warning(f'Variable {varname!r} in the subproject {subproject.subdir!r} is', 'not found' if var_dep is None else 'not a dependency object')
        return None
    return var_dep