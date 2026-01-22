from __future__ import annotations
import collections
import enum
import functools
import os
import itertools
import typing as T
from .. import build
from .. import coredata
from .. import dependencies
from .. import mesonlib
from .. import mlog
from ..compilers import SUFFIX_TO_LANG
from ..compilers.compilers import CompileCheckMode
from ..interpreterbase import (ObjectHolder, noPosargs, noKwargs,
from ..interpreterbase.decorators import ContainerTypeInfo, typed_kwargs, KwargInfo, typed_pos_args
from ..mesonlib import OptionKey
from .interpreterobjects import (extract_required_kwarg, extract_search_dirs)
from .type_checking import REQUIRED_KW, in_set_validator, NoneType
def _has_function_attribute_impl(self, attr: str, kwargs: T.Optional['ExtractRequired']=None) -> bool:
    """Common helper for function attribute testing."""
    logargs: TV_LoggableList = [f'Compiler for {self.compiler.get_display_language()} supports function attribute {attr}:']
    kwargs = kwargs or {'required': False}
    disabled, required, feature = extract_required_kwarg(kwargs, self.subproject, default=False)
    if disabled:
        logargs += ['skipped: feature', mlog.bold(feature), 'disabled']
        mlog.log(*logargs)
        return False
    had, cached = self.compiler.has_func_attribute(attr, self.environment)
    if required and (not had):
        logargs += ['not usable']
        raise InterpreterException(*logargs)
    logargs += [mlog.green('YES') if had else mlog.red('NO'), mlog.blue('(cached)') if cached else '']
    mlog.log(*logargs)
    return had