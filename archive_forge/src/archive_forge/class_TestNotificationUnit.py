from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
import time
import uuid
import boto
from gslib.cloud_api_delegator import CloudApiDelegator
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
from six import add_move, MovedModule
from six.moves import mock
class TestNotificationUnit(testcase.GsUtilUnitTestCase):

    @mock.patch.object(CloudApiDelegator, 'CreateNotificationConfig', autospec=True)
    def test_notification_splits_dash_m_value_correctly(self, mock_create_notification):
        bucket_uri = self.CreateBucket(bucket_name='foo_notification')
        stdout = self.RunCommand('notification', ['create', '-f', 'none', '-s', '-m', 'foo:bar:baz', suri(bucket_uri)], return_stdout=True)
        mock_create_notification.assert_called_once_with(mock.ANY, 'foo_notification', pubsub_topic=mock.ANY, payload_format=mock.ANY, custom_attributes={'foo': 'bar:baz'}, event_types=None, object_name_prefix=mock.ANY, provider=mock.ANY)