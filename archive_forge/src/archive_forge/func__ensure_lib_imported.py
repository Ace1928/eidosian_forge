import os
import sys
import shlex
import importlib
import subprocess
import pkg_resources
import pathlib
from typing import List, Type, Any, Union, Dict
from types import ModuleType
@classmethod
def _ensure_lib_imported(cls, library: str):
    clean_lib = cls.get_requirement(library, True)
    if not cls.is_imported(clean_lib):
        sys.modules[clean_lib] = importlib.import_module(clean_lib)
    return sys.modules[clean_lib]