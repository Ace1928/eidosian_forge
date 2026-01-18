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
def test_resident_set_size(self):
    max_memory = self.memory_limit(resource.RLIMIT_RSS)
    prlimit = processutils.ProcessLimits(resident_set_size=max_memory)
    self.check_limit(prlimit, 'RLIMIT_RSS', max_memory)