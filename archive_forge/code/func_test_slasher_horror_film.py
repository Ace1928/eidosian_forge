from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import sys
from unittest import mock
from gslib.exception import NO_URLS_MATCHED_PREFIX
from gslib.exception import NO_URLS_MATCHED_TARGET
import gslib.tests.testcase as testcase
from gslib.tests.testcase.base import MAX_BUCKET_LENGTH
from gslib.tests.testcase.integration_testcase import SkipForS3
import gslib.tests.util as util
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils import shim_util
from gslib.utils.retry_util import Retry
@SkipForS3('The boto lib used for S3 does not handle objects starting with slashes if we use V4 signature')
def test_slasher_horror_film(self):
    """Tests removing a bucket with objects that are filled with slashes."""
    bucket_uri = self.CreateVersionedBucket()
    ouri1 = self.CreateObject(bucket_uri=bucket_uri, object_name='h/e/l//lo', contents=b'Halloween')
    ouri2 = self.CreateObject(bucket_uri=bucket_uri, object_name='/h/e/l/l/o', contents=b'A Nightmare on Elm Street')
    ouri3 = self.CreateObject(bucket_uri=bucket_uri, object_name='//h//e/l//l/o', contents=b'Friday the 13th')
    ouri4 = self.CreateObject(bucket_uri=bucket_uri, object_name='//h//e//l//l//o', contents=b'I Know What You Did Last Summer')
    ouri5 = self.CreateObject(bucket_uri=bucket_uri, object_name='/', contents=b'Scream')
    ouri6 = self.CreateObject(bucket_uri=bucket_uri, object_name='//', contents=b"Child's Play")
    ouri7 = self.CreateObject(bucket_uri=bucket_uri, object_name='///', contents=b'The Prowler')
    ouri8 = self.CreateObject(bucket_uri=bucket_uri, object_name='////', contents=b'Black Christmas')
    ouri9 = self.CreateObject(bucket_uri=bucket_uri, object_name='everything/is/better/with/slashes///////', contents=b'Maniac')
    if self.multiregional_buckets:
        self.AssertNObjectsInBucket(bucket_uri, 9, versioned=True)
    objects_to_remove = ['%s#%s' % (suri(ouri1), urigen(ouri1)), '%s#%s' % (suri(ouri2), urigen(ouri2)), '%s#%s' % (suri(ouri3), urigen(ouri3)), '%s#%s' % (suri(ouri4), urigen(ouri4)), '%s#%s' % (suri(ouri5) + '/', urigen(ouri5)), '%s#%s' % (suri(ouri6) + '/', urigen(ouri6)), '%s#%s' % (suri(ouri7) + '/', urigen(ouri7)), '%s#%s' % (suri(ouri8) + '/', urigen(ouri8)), '%s#%s' % (suri(ouri9) + '/', urigen(ouri9))]
    self._RunRemoveCommandAndCheck(['-m', 'rm', '-r', suri(bucket_uri)], objects_to_remove=objects_to_remove, buckets_to_remove=[suri(bucket_uri)])