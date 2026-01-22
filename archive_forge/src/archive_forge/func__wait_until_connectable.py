import os
import time
from platform import system
from subprocess import DEVNULL
from subprocess import STDOUT
from subprocess import Popen
from typing_extensions import deprecated
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common import utils
def _wait_until_connectable(self, timeout=30):
    """Blocks until the extension is connectable in the firefox."""
    count = 0
    while not utils.is_connectable(self.profile.port):
        if self.process.poll():
            raise WebDriverException('The browser appears to have exited before we could connect. If you specified a log_file in the FirefoxBinary constructor, check it for details.')
        if count >= timeout:
            self.kill()
            raise WebDriverException(f"Can't load the profile. Possible firefox version mismatch. You must use GeckoDriver instead for Firefox 48+. Profile Dir: {self.profile.path} If you specified a log_file in the FirefoxBinary constructor, check it for details.")
        count += 1
        time.sleep(1)
    return True