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
def test_with_exit_code(self):
    exit_code = 0
    err = processutils.ProcessExecutionError(exit_code=exit_code)
    self.assertIn(str(exit_code), str(err))