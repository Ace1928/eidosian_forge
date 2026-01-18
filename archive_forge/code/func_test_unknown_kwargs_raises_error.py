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
def test_unknown_kwargs_raises_error(self):
    self.assertRaises(processutils.UnknownArgumentError, processutils.execute, '/usr/bin/env', 'true', this_is_not_a_valid_kwarg=True)