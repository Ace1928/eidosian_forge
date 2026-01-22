from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
import time
import uuid
import boto
from gslib.cloud_api_delegator import CloudApiDelegator
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
from six import add_move, MovedModule
from six.moves import mock
@Retry(AssertionError, tries=3, timeout_secs=5)
def _ListObjectChangeNotifications():
    stderr = self.RunGsUtil(['notification', 'list', '-o', suri(bucket_uri)], return_stderr=True)
    return stderr