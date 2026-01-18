from tests.unit import unittest
import time
import random
import boto.s3
from boto.compat import six, StringIO, urllib
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto.exception import S3ResponseError
def test_website_redirects(self):
    self.bucket.configure_website('index.html')
    key = self.bucket.new_key('redirect-key')
    self.assertTrue(key.set_redirect('http://www.amazon.com/'))
    self.assertEqual(key.get_redirect(), 'http://www.amazon.com/')
    self.assertTrue(key.set_redirect('http://aws.amazon.com/'))
    self.assertEqual(key.get_redirect(), 'http://aws.amazon.com/')