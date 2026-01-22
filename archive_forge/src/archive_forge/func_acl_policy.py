from mock import patch
import xml.dom.minidom
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.exception import BotoClientError
from boto.s3.connection import Location, S3Connection
from boto.s3.bucket import Bucket
from boto.s3.deletemarker import DeleteMarker
from boto.s3.key import Key
from boto.s3.multipart import MultiPartUpload
from boto.s3.prefix import Prefix
def acl_policy(self):
    return '<?xml version="1.0" encoding="UTF-8"?>\n        <AccessControlPolicy xmlns="http://s3.amazonaws.com/doc/2006-03-01/">\n          <Owner>\n            <ID>owner_id</ID>\n            <DisplayName>owner_display_name</DisplayName>\n          </Owner>\n          <AccessControlList>\n            <Grant>\n              <Grantee xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n                       xsi:type="CanonicalUser">\n                <ID>grantee_id</ID>\n                <DisplayName>grantee_display_name</DisplayName>\n              </Grantee>\n              <Permission>FULL_CONTROL</Permission>\n            </Grant>\n          </AccessControlList>\n        </AccessControlPolicy>'