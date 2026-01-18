from __future__ import annotations
import os
import typing as T
from ... import mesonlib
from ..compilers import CompileCheckMode
from .gnu import GnuLikeCompiler
from .visualstudio import VisualStudioLikeCompiler
Abstractions for ICL, the Intel compiler on Windows.