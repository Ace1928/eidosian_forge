import uuid
import fixtures
import flask
import flask_restful
import functools
from oslo_policy import policy
from oslo_serialization import jsonutils
from testtools import matchers
from keystone.common import context
from keystone.common import json_home
from keystone.common import rbac_enforcer
import keystone.conf
from keystone import exception
from keystone.server.flask import common as flask_common
from keystone.server.flask.request_processing import json_body
from keystone.tests.unit import rest
def _make_requests(self):
    path_base = '/arguments'
    api_prefix = self.restful_api_opts.get('api_url_prefix', '')
    blueprint_prefix = self.restful_api._blueprint_url_prefix.rstrip('/')
    url = ''.join([x for x in [blueprint_prefix, api_prefix, path_base] if x])
    headers = {'X-Auth-Token': self._get_token()}
    with self.test_client() as c:
        resp = c.get(url, headers=headers)
        self.assertEqual(_TestResourceWithCollectionInfo.wrap_collection([]), resp.json)
        unknown_id = uuid.uuid4().hex
        c.get('%s/%s' % (url, unknown_id), headers=headers, expected_status_code=404)
        c.head('%s/%s' % (url, unknown_id), headers=headers, expected_status_code=404)
        c.put('%s/%s' % (url, unknown_id), json={}, headers=headers, expected_status_code=404)
        c.patch('%s/%s' % (url, unknown_id), json={}, headers=headers, expected_status_code=404)
        c.delete('%s/%s' % (url, unknown_id), headers=headers, expected_status_code=404)
        new_argument_resource = {'testing': uuid.uuid4().hex}
        new_argument_resp = c.post(url, json=new_argument_resource, headers=headers).json['argument']
        new_argument2_resource = {'testing': uuid.uuid4().hex}
        new_argument2_resp = c.post(url, json=new_argument2_resource, headers=headers).json['argument']
        get_list_resp = c.get(url, headers=headers).json
        self.assertIn(new_argument_resp, get_list_resp['arguments'])
        self.assertIn(new_argument2_resp, get_list_resp['arguments'])
        get_resp = c.get('%s/%s' % (url, new_argument_resp['id']), headers=headers).json['argument']
        self.assertEqual(new_argument_resp, get_resp)
        head_resp = c.head('%s/%s' % (url, new_argument_resp['id']), headers=headers).data
        self.assertEqual(head_resp, b'')
        replacement_argument = {'new_arg': True, 'id': uuid.uuid4().hex}
        c.put('%s/%s' % (url, new_argument_resp['id']), headers=headers, json=replacement_argument, expected_status_code=400)
        replacement_argument.pop('id')
        c.put('%s/%s' % (url, new_argument_resp['id']), headers=headers, json=replacement_argument)
        put_resp = c.get('%s/%s' % (url, new_argument_resp['id']), headers=headers).json['argument']
        self.assertNotIn(new_argument_resp['testing'], put_resp)
        self.assertTrue(put_resp['new_arg'])
        get_replacement_resp = c.get('%s/%s' % (url, new_argument_resp['id']), headers=headers).json['argument']
        self.assertEqual(put_resp, get_replacement_resp)
        patch_ref = {'uuid': uuid.uuid4().hex}
        patch_resp = c.patch('%s/%s' % (url, new_argument_resp['id']), headers=headers, json=patch_ref).json['argument']
        self.assertTrue(patch_resp['new_arg'])
        self.assertEqual(patch_ref['uuid'], patch_resp['uuid'])
        get_patched_ref_resp = c.get('%s/%s' % (url, new_argument_resp['id']), headers=headers).json['argument']
        self.assertEqual(patch_resp, get_patched_ref_resp)
        c.delete('%s/%s' % (url, new_argument_resp['id']), headers=headers)
        c.get('%s/%s' % (url, new_argument_resp['id']), headers=headers, expected_status_code=404)