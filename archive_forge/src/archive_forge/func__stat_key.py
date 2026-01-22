from __future__ import annotations
import configparser
import logging
import os.path
from typing import Any
from flake8 import exceptions
from flake8.defaults import VALID_CODE_PREFIX
from flake8.options.manager import OptionManager
def _stat_key(s: str) -> tuple[int, int]:
    st = os.stat(s)
    return (st.st_ino, st.st_dev)