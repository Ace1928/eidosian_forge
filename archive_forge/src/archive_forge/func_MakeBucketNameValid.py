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
def MakeBucketNameValid(name):
    """Returns a copy of the given name with any invalid characters replaced.

  Args:
    name Union[str, unicode, bytes]: The bucket name to transform into a valid name.

  Returns:
    Union[str, unicode, bytes] The version of the bucket name containing only
      valid characters.
  """
    if isinstance(name, (six.text_type, six.binary_type)):
        return name.replace('_', '-').lower()
    else:
        raise TypeError('Unable to format name. Incorrect Type: {0}'.format(type(name)))