import logging
from typing import List
from pathlib import Path
from . import extract_and_copy_jar, get_wheel_android_arch
from .. import Config, find_pyside_modules
@wheel_shiboken.setter
def wheel_shiboken(self, wheel_shiboken: Path):
    self._wheel_shiboken = wheel_shiboken.resolve() if wheel_shiboken else None
    if self._wheel_shiboken:
        self.set_value('android', 'wheel_shiboken', str(self._wheel_shiboken))