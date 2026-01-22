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
def _ensure_binary_installed(cls, binary: str, flags: List[str]=None):
    return cls.install_binary(binary, flags)