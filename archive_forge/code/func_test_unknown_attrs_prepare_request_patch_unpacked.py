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
def test_unknown_attrs_prepare_request_patch_unpacked(self):

    class Test(resource.Resource):
        properties = resource.Body('properties')
        _store_unknown_attrs_as_properties = True
        commit_jsonpatch = True
    sot = Test.existing(**{'dummy': 'value', 'properties': 'a,b,c'})
    sot._update(**{'properties': {'dummy': 'new_value'}})
    request_body = sot._prepare_request(requires_id=False, patch=True).body
    self.assertDictEqual({u'path': u'/dummy', u'value': u'new_value', u'op': u'replace'}, request_body[0])