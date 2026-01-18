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
def test_patch_with_changed_fields(self):

    class Test(resource.Resource):
        allow_patch = True
        attr = resource.Body('attr')
        nested = resource.Body('renamed')
        other = resource.Body('other')
    sot = Test.existing(id=1, attr=42, nested={'dog': 'bark'})
    sot.attr = 'new'
    sot.patch(self.session, {'path': '/renamed/dog', 'op': 'remove'})
    expected = [{'path': '/attr', 'op': 'replace', 'value': 'new'}, {'path': '/renamed/dog', 'op': 'remove'}]
    self.session.patch.assert_called_once_with('/1', json=expected, headers=mock.ANY, microversion=None)