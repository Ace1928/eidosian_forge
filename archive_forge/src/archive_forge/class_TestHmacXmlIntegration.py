from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import boto
import os
import re
from gslib.commands import hmac
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
@SkipForS3('S3 does not have an equivalent API')
class TestHmacXmlIntegration(testcase.GsUtilIntegrationTestCase):
    """XML integration tests for the "hmac" command."""
    boto_config_hmac_auth_only = [('Credentials', 'gs_oauth2_refresh_token', None), ('Credentials', 'gs_service_client_id', None), ('Credentials', 'gs_service_key_file', None), ('Credentials', 'gs_service_key_file_password', None), ('Credentials', 'gs_access_key_id', 'dummykey'), ('Credentials', 'gs_secret_access_key', 'dummysecret')]

    def test_hmac_fails_for_xml(self):
        with SetBotoConfigForTest(self.boto_config_hmac_auth_only):
            for subcommand in ['create', 'delete', 'get', 'list', 'update']:
                command = ['hmac', subcommand]
                stderr = self.RunGsUtil(command, expected_status=1, return_stderr=True)
                self.assertIn('The "hmac" command can only be used with the GCS JSON API', stderr)