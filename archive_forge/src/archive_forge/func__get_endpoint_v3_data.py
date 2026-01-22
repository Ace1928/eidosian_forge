import collections
import os
import tempfile
import time
import urllib
import uuid
import fixtures
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from requests import structures
from requests_mock.contrib import fixture as rm_fixture
import openstack.cloud
import openstack.config as occ
import openstack.connection
from openstack.fixture import connection as os_fixture
from openstack.tests import base
from openstack.tests import fakes
def _get_endpoint_v3_data(self, service_id=None, region=None, url=None, interface=None, enabled=True):
    endpoint_id = uuid.uuid4().hex
    service_id = service_id or uuid.uuid4().hex
    region = region or uuid.uuid4().hex
    url = url or 'https://example.com/'
    interface = interface or uuid.uuid4().hex
    response = {'id': endpoint_id, 'service_id': service_id, 'region_id': region, 'interface': interface, 'url': url, 'enabled': enabled}
    request = response.copy()
    request.pop('id')
    return _EndpointDataV3(endpoint_id, service_id, interface, region, url, enabled, {'endpoint': response}, {'endpoint': request})