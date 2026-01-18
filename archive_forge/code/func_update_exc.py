import functools
import importlib
import importlib.util
import inspect
import itertools
import logging
import os
import pkgutil
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import (
import catalogue
import langcodes
import numpy
import srsly
import thinc
from catalogue import Registry, RegistryError
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from thinc.api import (
from thinc.api import compounding, decaying, fix_random_seed  # noqa: F401
from . import about
from .compat import CudaStream, cupy, importlib_metadata, is_windows
from .errors import OLD_MODEL_SHORTCUTS, Errors, Warnings
from .symbols import ORTH
def update_exc(base_exceptions: Dict[str, List[dict]], *addition_dicts) -> Dict[str, List[dict]]:
    """Update and validate tokenizer exceptions. Will overwrite exceptions.

    base_exceptions (Dict[str, List[dict]]): Base exceptions.
    *addition_dicts (Dict[str, List[dict]]): Exceptions to add to the base dict, in order.
    RETURNS (Dict[str, List[dict]]): Combined tokenizer exceptions.
    """
    exc = dict(base_exceptions)
    for additions in addition_dicts:
        for orth, token_attrs in additions.items():
            if not all((isinstance(attr[ORTH], str) for attr in token_attrs)):
                raise ValueError(Errors.E055.format(key=orth, orths=token_attrs))
            described_orth = ''.join((attr[ORTH] for attr in token_attrs))
            if orth != described_orth:
                raise ValueError(Errors.E056.format(key=orth, orths=described_orth))
        exc.update(additions)
    exc = expand_exc(exc, "'", '’')
    return exc