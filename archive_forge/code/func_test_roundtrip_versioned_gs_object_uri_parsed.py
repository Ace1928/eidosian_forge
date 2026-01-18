import boto
import tempfile
from boto.exception import InvalidUriError
from boto import storage_uri
from boto.compat import urllib
from boto.s3.keyfile import KeyFile
from tests.integration.s3.mock_storage_service import MockBucket
from tests.integration.s3.mock_storage_service import MockBucketStorageUri
from tests.integration.s3.mock_storage_service import MockConnection
from tests.unit import unittest
def test_roundtrip_versioned_gs_object_uri_parsed(self):
    uri_str = 'gs://bucket/obj#1359908801674000'
    uri = boto.storage_uri(uri_str, validate=False, suppress_consec_slashes=False)
    roundtrip_uri = boto.storage_uri(uri.uri, validate=False, suppress_consec_slashes=False)
    self.assertEqual(uri.uri, roundtrip_uri.uri)
    self.assertEqual(uri.is_version_specific, True)