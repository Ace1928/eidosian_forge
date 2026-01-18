import collections
import errno
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock
from oslotest import base as test_base
from oslo_concurrency.fixture import lockutils as fixtures
from oslo_concurrency import lockutils
from oslo_config import fixture as config
@mock.patch('logging.Logger.warning')
@mock.patch('os.remove', side_effect=OSError(errno.EPERM, None))
@mock.patch('oslo_concurrency.lockutils._get_lock_path')
def test_remove_lock_external_file_permission_error(self, path_mock, remove_mock, log_mock):
    lockutils.remove_external_lock_file(mock.sentinel.name, mock.sentinel.prefix, mock.sentinel.lock_path)
    path_mock.assert_called_once_with(mock.sentinel.name, mock.sentinel.prefix, mock.sentinel.lock_path)
    remove_mock.assert_called_once_with(path_mock.return_value)
    log_mock.assert_called()