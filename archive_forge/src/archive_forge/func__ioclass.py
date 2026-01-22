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
@property
def _ioclass(self):
    if self.isbinary:
        return io.BytesIO
    else:
        return io.StringIO