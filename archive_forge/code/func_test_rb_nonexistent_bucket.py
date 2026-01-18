from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
def test_rb_nonexistent_bucket(self):
    stderr = self.RunGsUtil(['rb', 'gs://%s' % self.nonexistent_bucket_name], return_stderr=True, expected_status=1)
    if self._use_gcloud_storage:
        self.assertIn('not found', stderr)
    else:
        self.assertIn('does not exist.', stderr)