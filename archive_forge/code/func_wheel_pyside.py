import logging
from typing import List
from pathlib import Path
from . import extract_and_copy_jar, get_wheel_android_arch
from .. import Config, find_pyside_modules
@wheel_pyside.setter
def wheel_pyside(self, wheel_pyside: Path):
    self._wheel_pyside = wheel_pyside.resolve() if wheel_pyside else None
    if self._wheel_pyside:
        self.set_value('android', 'wheel_pyside', str(self._wheel_pyside))