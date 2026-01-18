import hashlib
import io
from unittest import mock
import uuid
import boto3
import botocore
from botocore import exceptions as boto_exceptions
from botocore import stub
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers import s3
from glance_store import capabilities
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
def test_partial_get(self):
    loc = location.get_location_from_uri('s3://user:key@auth_address/glance/%s' % FAKE_UUID, conf=self.conf)
    self.assertRaises(exceptions.StoreRandomGetNotSupported, self.store.get, loc, chunk_size=1)