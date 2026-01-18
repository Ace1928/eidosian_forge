from tests.unit import AWSMockServiceTestCase
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.s3.lifecycle import Rule, Lifecycle, Transition
def test_parse_lifecycle_disabled(self):
    rule = self._get_bucket_lifecycle_config()[1]
    self.assertEqual(rule.status, 'Disabled')