from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import datetime
import locale
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import boto
from boto import config
from boto.exception import StorageResponseError
from boto.s3.deletemarker import DeleteMarker
from boto.storage_uri import BucketStorageUri
import gslib
from gslib.boto_translation import BotoTranslation
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import Preconditions
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.kms_api import KmsApi
from gslib.project_id import GOOG_PROJ_ID_HDR
from gslib.project_id import PopulateProjectId
from gslib.tests.testcase import base
import gslib.tests.util as util
from gslib.tests.util import InvokedFromParFile
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import RUN_S3_TESTS
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.tests.util import USING_JSON_API
import gslib.third_party.storage_apitools.storage_v1_messages as apitools_messages
from gslib.utils.constants import UTF8
from gslib.utils.encryption_helper import Base64Sha256FromBase64EncryptionKey
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
from gslib.utils.hashing_helper import Base64ToHexHash
from gslib.utils.metadata_util import CreateCustomMetadata
from gslib.utils.metadata_util import GetValueFromObjectCustomMetadata
from gslib.utils.posix_util import ATIME_ATTR
from gslib.utils.posix_util import GID_ATTR
from gslib.utils.posix_util import MODE_ATTR
from gslib.utils.posix_util import MTIME_ATTR
from gslib.utils.posix_util import UID_ATTR
from gslib.utils.retry_util import Retry
import six
from six.moves import range
def RunGsUtilTabCompletion(self, cmd, expected_results=None):
    """Runs the gsutil command in tab completion mode.

    Args:
      cmd: The command to run, as a list, e.g. ['cp', 'foo', 'bar']
      expected_results: The expected tab completion results for the given input.
    """
    cmd = [gslib.GSUTIL_PATH] + ['--testexceptiontraces'] + cmd
    if InvokedFromParFile():
        argcomplete_start_idx = 1
    else:
        argcomplete_start_idx = 2
        cmd = [str(sys.executable)] + cmd
    cmd_str = ' '.join(cmd)

    @Retry(AssertionError, tries=5, timeout_secs=1)
    def _RunTabCompletion():
        """Runs the tab completion operation with retries."""
        hacky_debugging = False
        results_string = None
        with tempfile.NamedTemporaryFile(delete=False) as tab_complete_result_file:
            if hacky_debugging:
                cmd_str_with_result_redirect = '{cs} 1>{fn} 2>{fn} 8>{fn} 9>{fn}'.format(cs=cmd_str, fn=tab_complete_result_file.name)
            else:
                cmd_str_with_result_redirect = '{cs} 8>{fn}'.format(cs=cmd_str, fn=tab_complete_result_file.name)
            env = os.environ.copy()
            env['_ARGCOMPLETE'] = str(argcomplete_start_idx)
            env['_ARGCOMPLETE_COMP_WORDBREAKS'] = '"\'@><=;|&(:'
            if 'COMP_WORDBREAKS' in env:
                env['_ARGCOMPLETE_COMP_WORDBREAKS'] = env['COMP_WORDBREAKS']
            env['COMP_LINE'] = cmd_str
            env['COMP_POINT'] = str(len(cmd_str))
            subprocess.call(cmd_str_with_result_redirect, env=env, shell=True)
            results_string = tab_complete_result_file.read().decode(locale.getpreferredencoding())
        if results_string:
            if hacky_debugging:
                print('---------------------------------------')
                print(results_string)
                print('---------------------------------------')
            results = results_string.split('\x0b')
        else:
            results = []
        self.assertEqual(results, expected_results)
    with SetBotoConfigForTest([('GSUtil', 'tab_completion_timeout', '120')]):
        _RunTabCompletion()