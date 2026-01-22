import os
import pathlib
import platform
import stat
import sys
from logging import getLogger
from typing import Union
def _check_bit(val: int, flag: int) -> bool:
    return bool(val & flag == flag)