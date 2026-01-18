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
@unittest.skipUnless(NOTIFICATION_URL, 'Test requires notification URL configuration.')
def test_watch_bucket(self):
    """Tests creating a notification channel on a bucket."""
    bucket_uri = self.CreateBucket()
    self.RunGsUtil(['notification', 'watchbucket', NOTIFICATION_URL, suri(bucket_uri)])
    identifier = str(uuid.uuid4())
    token = str(uuid.uuid4())
    stderr = self.RunGsUtil(['notification', 'watchbucket', '-i', identifier, '-t', token, NOTIFICATION_URL, suri(bucket_uri)], return_stderr=True)
    self.assertIn('token: %s' % token, stderr)
    self.assertIn('identifier: %s' % identifier, stderr)