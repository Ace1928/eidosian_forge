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
def _has_argument_impl(self, arguments: T.Union[str, T.List[str]], mode: _TestMode=_TestMode.COMPILER, kwargs: T.Optional['ExtractRequired']=None) -> bool:
    """Shared implementation for methods checking compiler and linker arguments."""
    if isinstance(arguments, str):
        arguments = [arguments]
    logargs: TV_LoggableList = ['Compiler for', self.compiler.get_display_language(), 'supports{}'.format(' link' if mode is _TestMode.LINKER else ''), 'arguments {}:'.format(' '.join(arguments))]
    kwargs = kwargs or {'required': False}
    disabled, required, feature = extract_required_kwarg(kwargs, self.subproject, default=False)
    if disabled:
        logargs += ['skipped: feature', mlog.bold(feature), 'disabled']
        mlog.log(*logargs)
        return False
    test = self.compiler.has_multi_link_arguments if mode is _TestMode.LINKER else self.compiler.has_multi_arguments
    result, cached = test(arguments, self.environment)
    if required and (not result):
        logargs += ['not usable']
        raise InterpreterException(*logargs)
    logargs += [mlog.green('YES') if result else mlog.red('NO'), mlog.blue('(cached)') if cached else '']
    mlog.log(*logargs)
    return result