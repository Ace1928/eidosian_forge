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
def _get_user_data(self, name=None, password=None, **kwargs):
    name = name or self.getUniqueString('username')
    password = password or self.getUniqueString('user_password')
    user_id = uuid.uuid4().hex
    response = {'name': name, 'id': user_id}
    request = {'name': name, 'password': password}
    if kwargs.get('domain_id'):
        kwargs['domain_id'] = uuid.UUID(kwargs['domain_id']).hex
        response['domain_id'] = kwargs.pop('domain_id')
        request['domain_id'] = response['domain_id']
    response['email'] = kwargs.pop('email', None)
    request['email'] = response['email']
    response['enabled'] = kwargs.pop('enabled', True)
    request['enabled'] = response['enabled']
    response['description'] = kwargs.pop('description', None)
    if response['description']:
        request['description'] = response['description']
    self.assertIs(0, len(kwargs), message='extra key-word args received on _get_user_data')
    return _UserData(user_id, password, name, response['email'], response['description'], response.get('domain_id'), response.get('enabled'), {'user': response}, {'user': request})