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
def test_number_processes(self):
    nprocs = self.soft_limit(resource.RLIMIT_NPROC, 1, 65535)
    prlimit = processutils.ProcessLimits(number_processes=nprocs)
    self.check_limit(prlimit, 'RLIMIT_NPROC', nprocs)