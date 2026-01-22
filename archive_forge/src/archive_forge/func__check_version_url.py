from unittest import mock
import fixtures
from keystoneauth1 import adapter
import logging
import requests
import testtools
from troveclient.apiclient import client
from troveclient import client as other_client
from troveclient import exceptions
from troveclient import service_catalog
import troveclient.v1.client
@mock.patch.object(other_client.HTTPClient, 'request', return_value=(200, "{'versions':[]}"))
def _check_version_url(self, management_url, version_url, mock_request):
    projectid = '25e469aa1848471b875e68cde6531bc5'
    instance = other_client.HTTPClient(user='user', password='password', projectid=projectid, auth_url='http://www.blah.com')
    instance.auth_token = 'foobar'
    instance.management_url = management_url % projectid
    mock_get_service_url = mock.Mock(return_value=instance.management_url)
    instance.get_service_url = mock_get_service_url
    instance.version = 'v2.0'
    instance.get('')
    mock_request.assert_called_once_with(instance.management_url, 'GET', headers=mock.ANY)
    mock_request.reset_mock()
    instance.get('/instances')
    url = instance.management_url + '/instances'
    mock_request.assert_called_once_with(url, 'GET', headers=mock.ANY)