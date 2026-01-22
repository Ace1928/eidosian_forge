import abc
import os
import sys
import pathlib
from contextlib import suppress
from typing import Union
def _native_reader(spec):
    reader = _available_reader(spec)
    return reader if hasattr(reader, 'files') else None