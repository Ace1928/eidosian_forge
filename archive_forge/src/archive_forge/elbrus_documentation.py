from __future__ import annotations
import functools
import os
import typing as T
import subprocess
import re
from .gnu import GnuLikeCompiler
from .gnu import gnu_optimization_args
from ...mesonlib import Popen_safe, OptionKey
Abstractions for the Elbrus family of compilers.