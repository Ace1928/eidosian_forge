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
def _SectionDictFromConfigList(boto_config_list):
    """Converts the input config list to a dict that is easy to write to a file.

  This is used to reset the boto config contents for a test instead of
  preserving the existing values.

  Args:
    boto_config_list: list of tuples of:
        (boto config section to set, boto config name to set, value to set)
        If value to set is None, no entry is created.

  Returns:
    Dictionary of {section: {keys: values}} for writing to the file.
  """
    sections = {}
    for config_entry in boto_config_list:
        section, key, value = (config_entry[0], config_entry[1], config_entry[2])
        if section not in sections:
            sections[section] = {}
        if value is not None:
            sections[section][key] = value
    return sections