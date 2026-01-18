from tests.unit import unittest
import time
import random
import boto.s3
from boto.compat import six, StringIO, urllib
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto.exception import S3ResponseError
def test_set_contents_with_md5(self):
    content = '01234567890123456789'
    sfp = StringIO(content)
    k = self.bucket.new_key('k')
    good_md5 = k.compute_md5(sfp)
    k.set_contents_from_file(sfp, md5=good_md5)
    kn = self.bucket.new_key('k')
    ks = kn.get_contents_as_string().decode('utf-8')
    self.assertEqual(ks, content)
    sfp.seek(5)
    k = self.bucket.new_key('k')
    good_md5 = k.compute_md5(sfp, size=5)
    k.set_contents_from_file(sfp, size=5, md5=good_md5)
    self.assertEqual(sfp.tell(), 10)
    kn = self.bucket.new_key('k')
    ks = kn.get_contents_as_string().decode('utf-8')
    self.assertEqual(ks, content[5:10])
    k = self.bucket.new_key('k')
    sfp.seek(0)
    hexdig, base64 = k.compute_md5(sfp)
    bad_md5 = (hexdig, base64[3:])
    try:
        k.set_contents_from_file(sfp, md5=bad_md5)
        self.fail('should fail with bad md5')
    except S3ResponseError:
        pass