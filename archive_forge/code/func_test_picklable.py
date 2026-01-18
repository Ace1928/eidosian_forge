import errno
import io
import logging
import multiprocessing
import os
import pickle
import resource
import socket
import stat
import subprocess
import sys
import tempfile
import time
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_concurrency import processutils
def test_picklable(self):
    exc = processutils.ProcessExecutionError(stdout='my stdout', stderr='my stderr', exit_code=42, cmd='my cmd', description='my description')
    exc_message = str(exc)
    exc = pickle.loads(pickle.dumps(exc))
    self.assertEqual('my stdout', exc.stdout)
    self.assertEqual('my stderr', exc.stderr)
    self.assertEqual(42, exc.exit_code)
    self.assertEqual('my cmd', exc.cmd)
    self.assertEqual('my description', exc.description)
    self.assertEqual(str(exc), exc_message)