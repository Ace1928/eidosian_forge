from tests.unit import unittest
from boto.sqs.message import MHMessage
from boto.sqs.message import RawMessage
from boto.sqs.message import Message
from boto.sqs.bigmessage import BigMessage
from boto.exception import SQSDecodeError
from nose.plugins.attrib import attr
class TestBigMessage(unittest.TestCase):

    @attr(sqs=True)
    def test_s3url_parsing(self):
        msg = BigMessage()
        bucket, key = msg._get_bucket_key('s3://foo')
        self.assertEquals(bucket, 'foo')
        self.assertEquals(key, None)
        bucket, key = msg._get_bucket_key('s3://foo/')
        self.assertEquals(bucket, 'foo')
        self.assertEquals(key, None)
        bucket, key = msg._get_bucket_key('s3://foo/bar')
        self.assertEquals(bucket, 'foo')
        self.assertEquals(key, 'bar')
        bucket, key = msg._get_bucket_key('s3://foo/bar/fie/baz')
        self.assertEquals(bucket, 'foo')
        self.assertEquals(key, 'bar/fie/baz')
        with self.assertRaises(SQSDecodeError) as context:
            bucket, key = msg._get_bucket_key('foo/bar')