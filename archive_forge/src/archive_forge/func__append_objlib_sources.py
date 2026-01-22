from __future__ import annotations
from functools import lru_cache
from os import environ
from pathlib import Path
import re
import typing as T
from .common import CMakeException, CMakeTarget, language_map, cmake_get_generator_args, check_cmake_args
from .fileapi import CMakeFileAPI
from .executor import CMakeExecutor
from .toolchain import CMakeToolchain, CMakeExecScope
from .traceparser import CMakeTraceParser
from .tracetargets import resolve_cmake_trace_targets
from .. import mlog, mesonlib
from ..mesonlib import MachineChoice, OrderedSet, path_is_in_root, relative_to_if_possible, OptionKey
from ..mesondata import DataFile
from ..compilers.compilers import assembler_suffixes, lang_suffixes, header_suffixes, obj_suffixes, lib_suffixes, is_header
from ..programs import ExternalProgram
from ..coredata import FORBIDDEN_TARGET_NAMES
from ..mparser import (
def _append_objlib_sources(self, tgt: 'ConverterTarget') -> None:
    self.includes += tgt.includes
    self.sources += tgt.sources
    self.generated += tgt.generated
    self.generated_ctgt += tgt.generated_ctgt
    self.includes = list(OrderedSet(self.includes))
    self.sources = list(OrderedSet(self.sources))
    self.generated = list(OrderedSet(self.generated))
    self.generated_ctgt = list(OrderedSet(self.generated_ctgt))
    for lang, opts in tgt.compile_opts.items():
        if lang not in self.compile_opts:
            self.compile_opts[lang] = []
        self.compile_opts[lang] += [x for x in opts if x not in self.compile_opts[lang]]