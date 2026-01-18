import itertools
import json
import logging
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_post_create_base_path(self):

    class Test(resource.Resource):
        service = self.service_name
        base_path = self.base_path
        allow_create = True
        create_method = 'POST'
    self._test_create(Test, requires_id=False, prepend_key=True, base_path='dummy')