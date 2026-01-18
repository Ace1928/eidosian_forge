import json
import os.path
import shutil
import tempfile
import unittest
import mock
import six
from apitools.base.py import credentials_lib
from apitools.base.py import util
def test_without_gflags(self):
    credentials_lib.FLAGS = None
    flags = credentials_lib._GetRunFlowFlags([])
    self.assertEqual(flags.auth_host_name, 'localhost')
    self.assertEqual(flags.auth_host_port, [8080, 8090])
    self.assertEqual(flags.logging_level, 'ERROR')
    self.assertEqual(flags.noauth_local_webserver, False)