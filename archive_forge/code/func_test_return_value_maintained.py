import collections
import errno
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock
from oslotest import base as test_base
from oslo_concurrency.fixture import lockutils as fixtures
from oslo_concurrency import lockutils
from oslo_config import fixture as config
def test_return_value_maintained(self):
    script = '\n'.join(['import sys', 'sys.exit(1)'])
    argv = ['', sys.executable, '-c', script]
    retval = lockutils._lock_wrapper(argv)
    self.assertEqual(1, retval)