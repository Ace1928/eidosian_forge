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
def test_works(self):
    out, err = processutils.ssh_execute(FakeSshConnection(0), 'ls')
    self.assertEqual('stdout', out)
    self.assertEqual('stderr', err)
    self.assertIsInstance(out, str)
    self.assertIsInstance(err, str)