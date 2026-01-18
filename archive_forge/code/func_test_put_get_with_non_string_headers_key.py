from tests.unit import unittest
import time
import random
import boto.s3
from boto.compat import six, StringIO, urllib
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto.exception import S3ResponseError
def test_put_get_with_non_string_headers_key(self):
    k = Key(self.bucket)
    k.key = 'foobar'
    body = 'This is a test of S3'
    k.set_contents_from_string(body)
    headers = {'Content-Length': 0}
    from_s3_key = self.bucket.get_key('foobar', headers=headers)
    self.assertEqual(from_s3_key.get_contents_as_string().decode('utf-8'), body)