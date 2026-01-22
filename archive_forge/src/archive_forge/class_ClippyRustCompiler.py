from __future__ import annotations
import functools
import subprocess, os.path
import textwrap
import re
import typing as T
from .. import coredata
from ..mesonlib import EnvironmentException, MesonException, Popen_safe_logged, OptionKey
from .compilers import Compiler, clike_debug_args
class ClippyRustCompiler(RustCompiler):
    """Clippy is a linter that wraps Rustc.

    This just provides us a different id
    """
    id = 'clippy-driver rustc'