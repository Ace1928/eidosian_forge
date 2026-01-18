from tests.unit import unittest
import time
import random
import boto.s3
from boto.compat import six, StringIO, urllib
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto.exception import S3ResponseError
def test_setting_date(self):
    key = self.bucket.new_key('test_date')
    key.set_metadata('date', '20130524T155935Z')
    key.set_contents_from_string('Some text here.')
    check = self.bucket.get_key('test_date')
    self.assertEqual(check.get_metadata('date'), u'20130524T155935Z')
    self.assertTrue('x-amz-meta-date' in check._get_remote_metadata())