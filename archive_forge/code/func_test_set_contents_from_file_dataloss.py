from tests.unit import unittest
import time
import random
import boto.s3
from boto.compat import six, StringIO, urllib
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto.exception import S3ResponseError
def test_set_contents_from_file_dataloss(self):
    content = 'abcde'
    sfp = StringIO()
    sfp.write(content)
    k = self.bucket.new_key('k')
    try:
        k.set_contents_from_file(sfp)
        self.fail('forgot to rewind so should fail.')
    except AttributeError:
        pass
    k.set_contents_from_file(sfp, rewind=True)
    self.assertEqual(k.size, 5)
    kn = self.bucket.new_key('k')
    ks = kn.get_contents_as_string().decode('utf-8')
    self.assertEqual(ks, content)
    sfp = StringIO()
    k = self.bucket.new_key('k')
    k.set_contents_from_file(sfp)
    self.assertEqual(k.size, 0)
    kn = self.bucket.new_key('k')
    ks = kn.get_contents_as_string().decode('utf-8')
    self.assertEqual(ks, '')