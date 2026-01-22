from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from contextlib import contextmanager
import functools
import locale
import logging
import os
import pkgutil
import posixpath
import re
import io
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
import six
from six.moves import urllib
from six.moves import cStringIO
import boto
import crcmod
import gslib
from gslib.kms_api import KmsApi
from gslib.project_id import PopulateProjectId
import mock_storage_service  # From boto/tests/integration/s3
from gslib.cloud_api import ResumableDownloadException
from gslib.cloud_api import ResumableUploadException
from gslib.lazy_wrapper import LazyWrapper
import gslib.tests as gslib_tests
from gslib.utils import posix_util
from gslib.utils.boto_util import UsingCrcmodExtension, HasUserSpecifiedGsHost
from gslib.utils.constants import UTF8
from gslib.utils.encryption_helper import Base64Sha256FromBase64EncryptionKey
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.unit_util import MakeHumanReadable
class TestParams(object):
    """Allows easier organization of test parameters.

  This class allows grouping of test parameters, which include args and kwargs
  to be used, as well as the expected result based on those arguments.

  For example, to test an Add function, one might do:

  params = TestParams(args=(1, 2, 3), expected=6)
  self.assertEqual(Add(*(params.args)), params.expected)
  """

    def __init__(self, args=None, kwargs=None, expected=None):
        self.args = tuple() if args is None else args
        self.kwargs = dict() if kwargs is None else kwargs
        self.expected = expected
        if not isinstance(args, (tuple, list)):
            raise TypeError('TestParam args must be a tuple or list.')
        if not isinstance(self.kwargs, dict):
            raise TypeError('TestParam kwargs must be a dict.')