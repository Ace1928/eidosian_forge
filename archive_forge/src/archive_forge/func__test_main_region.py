import io
import re
import sys
from unittest import mock
import fixtures
from keystoneauth1 import fixture
from testtools import matchers
from zunclient import api_versions
from zunclient import exceptions
import zunclient.shell
from zunclient.tests.unit import utils
@mock.patch('zunclient.client.Client')
def _test_main_region(self, command, expected_region_name, mock_client):
    self.shell(command)
    mock_client.assert_called_once_with(username='username', password='password', interface='publicURL', project_id=None, project_name='project_name', auth_url=self.AUTH_URL, service_type='container', region_name=expected_region_name, project_domain_id='', project_domain_name='', user_domain_id='', user_domain_name='', profile=None, endpoint_override=None, insecure=False, cacert=None, cert=None, key=None, version=api_versions.APIVersion('1.29'))