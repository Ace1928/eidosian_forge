from __future__ import annotations
import enum
import os.path
import string
import typing as T
from .. import coredata
from .. import mlog
from ..mesonlib import (
from .compilers import Compiler
def _to_host_compiler_options(self, options: 'KeyedOptionDictType') -> 'KeyedOptionDictType':
    """
        Convert an NVCC Option set to a host compiler's option set.
        """
    host_options = {key: options.get(key, opt) for key, opt in self.host_compiler.get_options().items()}
    std_key = OptionKey('std', machine=self.for_machine, lang=self.host_compiler.language)
    overrides = {std_key: 'none'}
    return coredata.OptionsView(host_options, overrides=overrides)