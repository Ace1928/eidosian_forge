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
def _get_subproject(self, subp_name: str) -> T.Optional[SubprojectHolder]:
    sub = self.interpreter.subprojects.get(subp_name)
    if sub and sub.found():
        return sub
    return None