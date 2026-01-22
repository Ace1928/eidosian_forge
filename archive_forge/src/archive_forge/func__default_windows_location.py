import os
import time
from platform import system
from subprocess import DEVNULL
from subprocess import STDOUT
from subprocess import Popen
from typing_extensions import deprecated
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common import utils
def _default_windows_location(self):
    program_files = [os.getenv('PROGRAMFILES', 'C:\\Program Files'), os.getenv('PROGRAMFILES(X86)', 'C:\\Program Files (x86)')]
    for path in program_files:
        binary_path = os.path.join(path, 'Mozilla Firefox\\firefox.exe')
        if os.access(binary_path, os.X_OK):
            return binary_path
    return ''