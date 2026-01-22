import os
import time
from platform import system
from subprocess import DEVNULL
from subprocess import STDOUT
from subprocess import Popen
from typing_extensions import deprecated
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common import utils
def _get_firefox_start_cmd(self):
    """Return the command to start firefox."""
    start_cmd = ''
    if self.platform == 'darwin':
        ffname = 'firefox'
        start_cmd = self.which(ffname)
        if not start_cmd:
            start_cmd = '/Applications/Firefox.app/Contents/MacOS/firefox'
        if not os.path.exists(start_cmd):
            start_cmd = os.path.expanduser('~') + start_cmd
    elif self.platform == 'windows':
        start_cmd = self._find_exe_in_registry() or self._default_windows_location()
    elif self.platform == 'java' and os.name == 'nt':
        start_cmd = self._default_windows_location()
    else:
        for ffname in ['firefox', 'iceweasel']:
            start_cmd = self.which(ffname)
            if start_cmd:
                break
        else:
            raise RuntimeError('Could not find firefox in your system PATH. Please specify the firefox binary location or install firefox')
    return start_cmd