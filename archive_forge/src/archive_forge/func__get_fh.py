import abc
import atexit
import contextlib
import logging
import os
import pathlib
import random
import tempfile
import time
import typing
import warnings
from . import constants, exceptions, portalocker
def _get_fh(self) -> typing.IO:
    """Get a new filehandle"""
    return open(self.filename, self.mode, **self.file_open_kwargs)