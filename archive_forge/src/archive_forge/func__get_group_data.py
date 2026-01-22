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
def _get_group_data(self, name=None, domain_id=None, description=None):
    group_id = uuid.uuid4().hex
    name = name or self.getUniqueString('groupname')
    domain_id = uuid.UUID(domain_id or uuid.uuid4().hex).hex
    response = {'id': group_id, 'name': name, 'domain_id': domain_id}
    request = {'name': name, 'domain_id': domain_id}
    if description is not None:
        response['description'] = description
        request['description'] = description
    return _GroupData(group_id, name, domain_id, description, {'group': response}, {'group': request})