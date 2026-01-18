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
def test_process_input_with_string(self):
    code = ';'.join(('import sys', 'print(len(sys.stdin.readlines()))'))
    args = [sys.executable, '-c', code]
    input = '\n'.join(['foo', 'bar', 'baz'])
    stdout, stderr = processutils.execute(*args, process_input=input)
    self.assertEqual('3', stdout.rstrip())