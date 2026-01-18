import os
import unittest
import time
from boto.compat import StringIO
import mock
import boto
from boto.s3.connection import S3Connection
def test_upload_part_by_size(self):
    key_name = 'k'
    contents = '01234567890123456789'
    sfp = StringIO(contents)
    mpu = self.bucket.initiate_multipart_upload(key_name)
    mpu.upload_part_from_file(sfp, part_num=1, size=5)
    mpu.upload_part_from_file(sfp, part_num=2, size=5)
    mpu.upload_part_from_file(sfp, part_num=3, size=5)
    mpu.upload_part_from_file(sfp, part_num=4, size=5)
    sfp.close()
    etags = {}
    pn = 0
    for part in mpu:
        pn += 1
        self.assertEqual(5, part.size)
        etags[pn] = part.etag
    self.assertEqual(pn, 4)
    self.assertEqual(etags[1], etags[3])
    self.assertEqual(etags[2], etags[4])
    self.assertNotEqual(etags[1], etags[2])
    mpu.cancel_upload()