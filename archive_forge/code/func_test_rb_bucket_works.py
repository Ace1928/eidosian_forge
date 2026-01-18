from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
def test_rb_bucket_works(self):
    bucket_uri = self.CreateBucket()
    self.RunGsUtil(['rb', suri(bucket_uri)])
    stderr = self.RunGsUtil(['ls', '-Lb', 'gs://%s' % self.nonexistent_bucket_name], return_stderr=True, expected_status=1)
    self.assertIn('404', stderr)