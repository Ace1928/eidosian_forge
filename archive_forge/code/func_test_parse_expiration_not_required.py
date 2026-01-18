from tests.unit import AWSMockServiceTestCase
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.s3.lifecycle import Rule, Lifecycle, Transition
def test_parse_expiration_not_required(self):
    rule = self._get_bucket_lifecycle_config()[2]
    self.assertIsNone(rule.expiration)