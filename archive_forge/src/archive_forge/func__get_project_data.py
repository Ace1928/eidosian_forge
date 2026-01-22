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
def _get_project_data(self, project_name=None, enabled=None, domain_id=None, description=None, v3=True, project_id=None, parent_id=None):
    project_name = project_name or self.getUniqueString('projectName')
    project_id = uuid.UUID(project_id or uuid.uuid4().hex).hex
    if parent_id:
        parent_id = uuid.UUID(parent_id).hex
    response = {'id': project_id, 'name': project_name}
    request = {'name': project_name}
    domain_id = domain_id or uuid.uuid4().hex if v3 else None
    if domain_id:
        request['domain_id'] = domain_id
        response['domain_id'] = domain_id
    if enabled is not None:
        enabled = bool(enabled)
        response['enabled'] = enabled
        request['enabled'] = enabled
    if parent_id:
        request['parent_id'] = parent_id
        response['parent_id'] = parent_id
    response.setdefault('enabled', True)
    request.setdefault('enabled', True)
    if description:
        response['description'] = description
        request['description'] = description
    request.setdefault('description', None)
    return _ProjectData(project_id, project_name, enabled, domain_id, description, parent_id, {'project': response}, {'project': request})