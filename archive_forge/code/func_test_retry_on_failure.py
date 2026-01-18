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
def test_retry_on_failure(self):
    fd, tmpfilename = tempfile.mkstemp()
    _, tmpfilename2 = tempfile.mkstemp()
    try:
        fp = os.fdopen(fd, 'w+')
        fp.write('#!/bin/sh\n# If stdin fails to get passed during one of the runs, make a note.\nif ! grep -q foo\nthen\n    echo \'failure\' > "$1"\nfi\n# If stdin has failed to get passed during this or a previous run, exit early.\nif grep failure "$1"\nthen\n    exit 1\nfi\nruns="$(cat $1)"\nif [ -z "$runs" ]\nthen\n    runs=0\nfi\nruns=$(($runs + 1))\necho $runs > "$1"\nexit 1\n')
        fp.close()
        os.chmod(tmpfilename, 493)
        self.assertRaises(processutils.ProcessExecutionError, processutils.execute, tmpfilename, tmpfilename2, attempts=10, process_input=b'foo', delay_on_retry=False)
        fp = open(tmpfilename2, 'r')
        runs = fp.read()
        fp.close()
        self.assertNotEqual('failure', 'stdin did not always get passed correctly', runs.strip())
        runs = int(runs.strip())
        self.assertEqual(10, runs, 'Ran %d times instead of 10.' % (runs,))
    finally:
        os.unlink(tmpfilename)
        os.unlink(tmpfilename2)