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
def GetGsutilCommand(raw_command, force_gsutil=False):
    """Adds config options to a list of strings defining a gsutil subcommand."""
    if force_gsutil:
        use_gcloud_storage = False
    else:
        use_gcloud_storage = boto.config.getbool('GSUtil', 'use_gcloud_storage', False)
    gcloud_storage_setting = ['-o', 'GSUtil:use_gcloud_storage={}'.format(use_gcloud_storage), '-o', 'GSUtil:hidden_shim_mode=no_fallback']
    gsutil_command = [gslib.GSUTIL_PATH, '--testexceptiontraces', '-o', 'GSUtil:default_project_id=' + PopulateProjectId()] + gcloud_storage_setting + raw_command
    if not InvokedFromParFile():
        gsutil_command_with_executable_path = [str(sys.executable)] + gsutil_command
    else:
        gsutil_command_with_executable_path = gsutil_command
    return [six.ensure_str(part) for part in gsutil_command_with_executable_path]