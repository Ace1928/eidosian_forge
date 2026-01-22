import logging
from typing import List
from pathlib import Path
from . import extract_and_copy_jar, get_wheel_android_arch
from .. import Config, find_pyside_modules
def _find_and_set_arch(self):
    """Find architecture from wheel name
        """
    self.arch = get_wheel_android_arch(wheel=self.wheel_pyside)
    if not self.arch:
        raise RuntimeError('[DEPLOY] PySide wheel corrupted. Wheel name should end withplatform name')