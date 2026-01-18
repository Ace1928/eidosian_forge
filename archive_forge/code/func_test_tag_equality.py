from tests.unit import AWSMockServiceTestCase
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.s3.tagging import Tag
def test_tag_equality(self):
    t1 = Tag('foo', 'bar')
    t2 = Tag('foo', 'bar')
    t3 = Tag('foo', 'baz')
    t4 = Tag('baz', 'bar')
    self.assertEqual(t1, t2)
    self.assertNotEqual(t1, t3)
    self.assertNotEqual(t1, t4)