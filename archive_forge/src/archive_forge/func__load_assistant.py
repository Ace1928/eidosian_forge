import glob
import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any, Dict
from pkg_resources import parse_requirements
from setuptools import find_packages
def _load_assistant() -> ModuleType:
    location = os.path.join(_PROJECT_ROOT, '.actions', 'assistant.py')
    return _load_py_module('assistant', location)