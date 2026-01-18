import io
import re
import functools
import inspect
import os
import sys
import numbers
import warnings
from pathlib import Path, PurePath
from typing import (
from ase.atoms import Atoms
from importlib import import_module
from ase.parallel import parallel_function, parallel_generator
def wrap_read_function(read, filename, index=None, **kwargs):
    """Convert read-function to generator."""
    if index is None:
        yield read(filename, **kwargs)
    else:
        for atoms in read(filename, index, **kwargs):
            yield atoms