from __future__ import annotations
import os
import shutil
import typing as T
from ... import mesonlib
from ...linkers.linkers import AppleDynamicLinker, ClangClDynamicLinker, LLVMDynamicLinker, GnuGoldDynamicLinker, \
from ...mesonlib import OptionKey
from ..compilers import CompileCheckMode
from .gnu import GnuLikeCompiler
Abstractions for the LLVM/Clang compiler family.