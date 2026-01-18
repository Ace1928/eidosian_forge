import os
import sys
import subprocess
import time
from pecan.compat import urlopen, URLError
from pecan.tests import PecanTestCase
import unittest
def test_serve_from_config(self):
    proc = subprocess.Popen([os.path.join(self.bin, 'uwsgi'), '--http-socket', ':8080', '--venv', sys.prefix, '--pecan', 'testing123/config.py'])
    self.poll_http('uwsgi', proc, 8080)