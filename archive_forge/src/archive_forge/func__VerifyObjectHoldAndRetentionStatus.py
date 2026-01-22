from __future__ import absolute_import
import datetime
import re
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
def _VerifyObjectHoldAndRetentionStatus(self, bucket_uri, object_uri, temporary_hold=None, event_based_hold=None, retention_period=None):
    object_metadata = self.json_api.GetObjectMetadata(bucket_uri.bucket_name, object_uri.object_name, fields=['timeCreated', 'temporaryHold', 'eventBasedHold', 'retentionExpirationTime'])
    if temporary_hold is None:
        self.assertEqual(object_metadata.temporaryHold, None)
    else:
        self.assertEqual(object_metadata.temporaryHold, temporary_hold)
    if event_based_hold is None:
        self.assertEqual(object_metadata.eventBasedHold, None)
    else:
        self.assertEqual(object_metadata.eventBasedHold, event_based_hold)
    if retention_period is None:
        self.assertEqual(object_metadata.retentionExpirationTime, None)
    elif event_based_hold is False:
        retention_policy = self.json_api.GetBucket(bucket_uri.bucket_name, fields=['retentionPolicy']).retentionPolicy
        time_delta = datetime.timedelta(0, retention_policy.retentionPeriod)
        expected_expiration_time = object_metadata.timeCreated + time_delta
        if event_based_hold is None:
            self.assertEqual(object_metadata.retentionExpirationTime, expected_expiration_time)
        else:
            self.assertGreater(object_metadata.retentionExpirationTime, expected_expiration_time)