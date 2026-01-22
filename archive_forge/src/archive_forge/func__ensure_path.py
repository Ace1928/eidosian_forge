import configparser
import os
import sys
import logging
import typing
from . import backend, credentials
from .util import platform_ as platform
from .backends import fail
def _ensure_path(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return path