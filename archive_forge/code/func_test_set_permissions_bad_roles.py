import re
import sys
import copy
import json
import unittest
import email.utils
from io import BytesIO
from unittest import mock
from unittest.mock import Mock, PropertyMock
import pytest
from libcloud.test import StorageMockHttp
from libcloud.utils.py3 import StringIO, httplib
from libcloud.common.types import InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_GOOGLE_STORAGE_PARAMS
from libcloud.common.google import GoogleAuthType
from libcloud.storage.drivers import google_storage
from libcloud.test.file_fixtures import StorageFileFixtures
from libcloud.test.storage.test_s3 import S3Tests, S3MockHttp
from libcloud.test.common.test_google import GoogleTestCase
def test_set_permissions_bad_roles(self):
    mock_request = mock.Mock()
    self.driver.json_connection.request = mock_request
    self.assertRaises(ValueError, self.driver.ex_set_permissions, 'bucket', 'object')
    self.assertRaises(ValueError, self.driver.ex_set_permissions, 'bucket')
    mock_request.assert_not_called()
    self.assertRaises(ValueError, self.driver.ex_set_permissions, 'bucket', 'object', role=google_storage.ContainerPermissions.OWNER)
    mock_request.assert_not_called()
    self.assertRaises(ValueError, self.driver.ex_set_permissions, 'bucket', role=google_storage.ObjectPermissions.OWNER)
    mock_request.assert_not_called()