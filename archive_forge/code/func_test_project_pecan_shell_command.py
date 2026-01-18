import os
import sys
import subprocess
import time
from pecan.compat import urlopen, URLError
from pecan.tests import PecanTestCase
import unittest
def test_project_pecan_shell_command(self):
    proc = subprocess.Popen([os.path.join(self.bin, 'pecan'), 'shell', 'testing123/config.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    self.poll(proc)
    out, _ = proc.communicate(b'{"model" : model, "conf" : conf, "app" : app}')
    assert 'testing123.model' in out.decode(), out
    assert 'Config(' in out.decode(), out
    assert 'webtest.app.TestApp' in out.decode(), out
    try:
        proc.terminate()
    except:
        pass