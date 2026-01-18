import threading
import uuid
from os_win import exceptions
from os_win.tests.functional import test_base
from os_win.utils import processutils
def test_release_unacquired_mutex(self):
    self.assertRaises(exceptions.Win32Exception, self._mutex.release)