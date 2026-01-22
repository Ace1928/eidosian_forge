from __future__ import annotations
from .base import ExternalDependency, DependencyException, DependencyTypeName
from .pkgconfig import PkgConfigDependency
from ..mesonlib import (Popen_safe, OptionKey, join_args, version_compare)
from ..programs import ExternalProgram
from .. import mlog
import re
import os
import json
import typing as T
def _call_compbin(self, args: T.List[str], env: T.Optional[T.Dict[str, str]]=None) -> T.Tuple[int, str, str]:
    p, out, err = Popen_safe(self.compiler.get_exelist() + args, env=env)
    return (p.returncode, out.strip(), err.strip())