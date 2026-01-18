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
def test_with_env_variables(self):
    env_vars = {'SUPER_UNIQUE_VAR': 'The answer is 42'}
    out, err = processutils.execute('/usr/bin/env', env_variables=env_vars)
    self.assertIsInstance(out, str)
    self.assertIsInstance(err, str)
    self.assertIn('SUPER_UNIQUE_VAR=The answer is 42', out)