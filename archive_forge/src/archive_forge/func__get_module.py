import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import warnings
from collections import OrderedDict
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any, Tuple, Union
from packaging import version
from . import logging
def _get_module(self, module_name: str):
    try:
        return importlib.import_module('.' + module_name, self.__name__)
    except Exception as e:
        raise RuntimeError(f'Failed to import {self.__name__}.{module_name} because of the following error (look up to see its traceback):\n{e}') from e