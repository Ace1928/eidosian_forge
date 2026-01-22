import datetime
import hashlib
import http.client as http
import os
import requests
from unittest import mock
import uuid
from castellan.common import exception as castellan_exception
import glance_store as store
from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils import fixture
import testtools
import webob
import webob.exc
import glance.api.v2.image_actions
import glance.api.v2.images
from glance.common import exception
from glance.common import store_utils
from glance.common import timeutils
from glance import domain
import glance.notifier
import glance.schema
from glance.tests.unit import base
from glance.tests.unit.keymgr import fake as fake_keymgr
import glance.tests.unit.utils as unit_test_utils
from glance.tests.unit.v2 import test_tasks_resource
import glance.tests.utils as test_utils
class TestImagesDeserializer(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImagesDeserializer, self).setUp()
        self.deserializer = glance.api.v2.images.RequestDeserializer()

    def test_create_minimal(self):
        request = unit_test_utils.get_fake_request()
        request.body = jsonutils.dump_as_bytes({})
        output = self.deserializer.create(request)
        expected = {'image': {}, 'extra_properties': {}, 'tags': []}
        self.assertEqual(expected, output)

    def test_create_invalid_id(self):
        request = unit_test_utils.get_fake_request()
        request.body = jsonutils.dump_as_bytes({'id': 'gabe'})
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.create, request)

    def test_create_id_to_image_id(self):
        request = unit_test_utils.get_fake_request()
        request.body = jsonutils.dump_as_bytes({'id': UUID4})
        output = self.deserializer.create(request)
        expected = {'image': {'image_id': UUID4}, 'extra_properties': {}, 'tags': []}
        self.assertEqual(expected, output)

    def test_create_no_body(self):
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.create, request)

    def test_create_full(self):
        request = unit_test_utils.get_fake_request()
        request.body = jsonutils.dump_as_bytes({'id': UUID3, 'name': 'image-1', 'visibility': 'public', 'tags': ['one', 'two'], 'container_format': 'ami', 'disk_format': 'ami', 'min_ram': 128, 'min_disk': 10, 'foo': 'bar', 'protected': True})
        output = self.deserializer.create(request)
        properties = {'image_id': UUID3, 'name': 'image-1', 'visibility': 'public', 'container_format': 'ami', 'disk_format': 'ami', 'min_ram': 128, 'min_disk': 10, 'protected': True}
        self.maxDiff = None
        expected = {'image': properties, 'extra_properties': {'foo': 'bar'}, 'tags': ['one', 'two']}
        self.assertEqual(expected, output)

    def test_create_invalid_property_key(self):
        request = unit_test_utils.get_fake_request()
        request.body = jsonutils.dump_as_bytes({'id': UUID3, 'name': 'image-1', 'visibility': 'public', 'tags': ['one', 'two'], 'container_format': 'ami', 'disk_format': 'ami', 'min_ram': 128, 'min_disk': 10, 'f' * 256: 'bar', 'protected': True})
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.create, request)

    def test_create_readonly_attributes_forbidden(self):
        bodies = [{'direct_url': 'http://example.com'}, {'self': 'http://example.com'}, {'file': 'http://example.com'}, {'schema': 'http://example.com'}, {'os_glance_foo': 'foo'}]
        for body in bodies:
            request = unit_test_utils.get_fake_request()
            request.body = jsonutils.dump_as_bytes(body)
            self.assertRaises(webob.exc.HTTPForbidden, self.deserializer.create, request)

    def _get_fake_patch_request(self, content_type_minor_version=1):
        request = unit_test_utils.get_fake_request()
        template = 'application/openstack-images-v2.%d-json-patch'
        request.content_type = template % content_type_minor_version
        return request

    def test_update_empty_body(self):
        request = self._get_fake_patch_request()
        request.body = jsonutils.dump_as_bytes([])
        output = self.deserializer.update(request)
        expected = {'changes': []}
        self.assertEqual(expected, output)

    def test_update_unsupported_content_type(self):
        request = unit_test_utils.get_fake_request()
        request.content_type = 'application/json-patch'
        request.body = jsonutils.dump_as_bytes([])
        try:
            self.deserializer.update(request)
        except webob.exc.HTTPUnsupportedMediaType as e:
            accept_patch = ['application/openstack-images-v2.1-json-patch', 'application/openstack-images-v2.0-json-patch']
            expected = ', '.join(sorted(accept_patch))
            self.assertEqual(expected, e.headers['Accept-Patch'])
        else:
            self.fail('Did not raise HTTPUnsupportedMediaType')

    def test_update_body_not_a_list(self):
        bodies = [{'op': 'add', 'path': '/someprop', 'value': 'somevalue'}, 'just some string', 123, True, False, None]
        for body in bodies:
            request = self._get_fake_patch_request()
            request.body = jsonutils.dump_as_bytes(body)
            self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.update, request)

    def test_update_invalid_changes(self):
        changes = [['a', 'list', 'of', 'stuff'], 'just some string', 123, True, False, None, {'op': 'invalid', 'path': '/name', 'value': 'fedora'}]
        for change in changes:
            request = self._get_fake_patch_request()
            request.body = jsonutils.dump_as_bytes([change])
            self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.update, request)

    def test_update_invalid_validation_data(self):
        request = self._get_fake_patch_request()
        changes = [{'op': 'add', 'path': '/locations/0', 'value': {'url': 'http://localhost/fake', 'metadata': {}}}]
        changes[0]['value']['validation_data'] = {'os_hash_algo': 'sha512', 'os_hash_value': MULTIHASH1, 'checksum': CHKSUM}
        request.body = jsonutils.dump_as_bytes(changes)
        self.deserializer.update(request)
        changes[0]['value']['validation_data'] = {'os_hash_algo': 'sha512', 'os_hash_value': MULTIHASH1, 'checksum': CHKSUM, 'bogus_key': 'bogus_value'}
        request.body = jsonutils.dump_as_bytes(changes)
        self.assertRaisesRegex(webob.exc.HTTPBadRequest, 'Additional properties are not allowed', self.deserializer.update, request)
        changes[0]['value']['validation_data'] = {'checksum': CHKSUM}
        request.body = jsonutils.dump_as_bytes(changes)
        self.assertRaisesRegex(webob.exc.HTTPBadRequest, 'os_hash.* is a required property', self.deserializer.update, request)

    def test_update(self):
        request = self._get_fake_patch_request()
        body = [{'op': 'replace', 'path': '/name', 'value': 'fedora'}, {'op': 'replace', 'path': '/tags', 'value': ['king', 'kong']}, {'op': 'replace', 'path': '/foo', 'value': 'bar'}, {'op': 'add', 'path': '/bebim', 'value': 'bap'}, {'op': 'remove', 'path': '/sparks'}, {'op': 'add', 'path': '/locations/-', 'value': {'url': 'scheme3://path3', 'metadata': {}}}, {'op': 'add', 'path': '/locations/10', 'value': {'url': 'scheme4://path4', 'metadata': {}}}, {'op': 'remove', 'path': '/locations/2'}, {'op': 'replace', 'path': '/locations', 'value': []}, {'op': 'replace', 'path': '/locations', 'value': [{'url': 'scheme5://path5', 'metadata': {}}, {'url': 'scheme6://path6', 'metadata': {}}]}]
        request.body = jsonutils.dump_as_bytes(body)
        output = self.deserializer.update(request)
        expected = {'changes': [{'json_schema_version': 10, 'op': 'replace', 'path': ['name'], 'value': 'fedora'}, {'json_schema_version': 10, 'op': 'replace', 'path': ['tags'], 'value': ['king', 'kong']}, {'json_schema_version': 10, 'op': 'replace', 'path': ['foo'], 'value': 'bar'}, {'json_schema_version': 10, 'op': 'add', 'path': ['bebim'], 'value': 'bap'}, {'json_schema_version': 10, 'op': 'remove', 'path': ['sparks']}, {'json_schema_version': 10, 'op': 'add', 'path': ['locations', '-'], 'value': {'url': 'scheme3://path3', 'metadata': {}}}, {'json_schema_version': 10, 'op': 'add', 'path': ['locations', '10'], 'value': {'url': 'scheme4://path4', 'metadata': {}}}, {'json_schema_version': 10, 'op': 'remove', 'path': ['locations', '2']}, {'json_schema_version': 10, 'op': 'replace', 'path': ['locations'], 'value': []}, {'json_schema_version': 10, 'op': 'replace', 'path': ['locations'], 'value': [{'url': 'scheme5://path5', 'metadata': {}}, {'url': 'scheme6://path6', 'metadata': {}}]}]}
        self.assertEqual(expected, output)

    def test_update_v2_0_compatibility(self):
        request = self._get_fake_patch_request(content_type_minor_version=0)
        body = [{'replace': '/name', 'value': 'fedora'}, {'replace': '/tags', 'value': ['king', 'kong']}, {'replace': '/foo', 'value': 'bar'}, {'add': '/bebim', 'value': 'bap'}, {'remove': '/sparks'}, {'add': '/locations/-', 'value': {'url': 'scheme3://path3', 'metadata': {}}}, {'add': '/locations/10', 'value': {'url': 'scheme4://path4', 'metadata': {}}}, {'remove': '/locations/2'}, {'replace': '/locations', 'value': []}, {'replace': '/locations', 'value': [{'url': 'scheme5://path5', 'metadata': {}}, {'url': 'scheme6://path6', 'metadata': {}}]}]
        request.body = jsonutils.dump_as_bytes(body)
        output = self.deserializer.update(request)
        expected = {'changes': [{'json_schema_version': 4, 'op': 'replace', 'path': ['name'], 'value': 'fedora'}, {'json_schema_version': 4, 'op': 'replace', 'path': ['tags'], 'value': ['king', 'kong']}, {'json_schema_version': 4, 'op': 'replace', 'path': ['foo'], 'value': 'bar'}, {'json_schema_version': 4, 'op': 'add', 'path': ['bebim'], 'value': 'bap'}, {'json_schema_version': 4, 'op': 'remove', 'path': ['sparks']}, {'json_schema_version': 4, 'op': 'add', 'path': ['locations', '-'], 'value': {'url': 'scheme3://path3', 'metadata': {}}}, {'json_schema_version': 4, 'op': 'add', 'path': ['locations', '10'], 'value': {'url': 'scheme4://path4', 'metadata': {}}}, {'json_schema_version': 4, 'op': 'remove', 'path': ['locations', '2']}, {'json_schema_version': 4, 'op': 'replace', 'path': ['locations'], 'value': []}, {'json_schema_version': 4, 'op': 'replace', 'path': ['locations'], 'value': [{'url': 'scheme5://path5', 'metadata': {}}, {'url': 'scheme6://path6', 'metadata': {}}]}]}
        self.assertEqual(expected, output)

    def test_update_base_attributes(self):
        request = self._get_fake_patch_request()
        body = [{'op': 'replace', 'path': '/name', 'value': 'fedora'}, {'op': 'replace', 'path': '/visibility', 'value': 'public'}, {'op': 'replace', 'path': '/tags', 'value': ['king', 'kong']}, {'op': 'replace', 'path': '/protected', 'value': True}, {'op': 'replace', 'path': '/container_format', 'value': 'bare'}, {'op': 'replace', 'path': '/disk_format', 'value': 'raw'}, {'op': 'replace', 'path': '/min_ram', 'value': 128}, {'op': 'replace', 'path': '/min_disk', 'value': 10}, {'op': 'replace', 'path': '/locations', 'value': []}, {'op': 'replace', 'path': '/locations', 'value': [{'url': 'scheme5://path5', 'metadata': {}}, {'url': 'scheme6://path6', 'metadata': {}}]}]
        request.body = jsonutils.dump_as_bytes(body)
        output = self.deserializer.update(request)
        expected = {'changes': [{'json_schema_version': 10, 'op': 'replace', 'path': ['name'], 'value': 'fedora'}, {'json_schema_version': 10, 'op': 'replace', 'path': ['visibility'], 'value': 'public'}, {'json_schema_version': 10, 'op': 'replace', 'path': ['tags'], 'value': ['king', 'kong']}, {'json_schema_version': 10, 'op': 'replace', 'path': ['protected'], 'value': True}, {'json_schema_version': 10, 'op': 'replace', 'path': ['container_format'], 'value': 'bare'}, {'json_schema_version': 10, 'op': 'replace', 'path': ['disk_format'], 'value': 'raw'}, {'json_schema_version': 10, 'op': 'replace', 'path': ['min_ram'], 'value': 128}, {'json_schema_version': 10, 'op': 'replace', 'path': ['min_disk'], 'value': 10}, {'json_schema_version': 10, 'op': 'replace', 'path': ['locations'], 'value': []}, {'json_schema_version': 10, 'op': 'replace', 'path': ['locations'], 'value': [{'url': 'scheme5://path5', 'metadata': {}}, {'url': 'scheme6://path6', 'metadata': {}}]}]}
        self.assertEqual(expected, output)

    def test_update_disallowed_attributes(self):
        samples = {'direct_url': '/a/b/c/d', 'self': '/e/f/g/h', 'file': '/e/f/g/h/file', 'schema': '/i/j/k'}
        for key, value in samples.items():
            request = self._get_fake_patch_request()
            body = [{'op': 'replace', 'path': '/%s' % key, 'value': value}]
            request.body = jsonutils.dump_as_bytes(body)
            try:
                self.deserializer.update(request)
            except webob.exc.HTTPForbidden:
                pass
            else:
                self.fail('Updating %s did not result in HTTPForbidden' % key)

    def test_update_readonly_attributes(self):
        samples = {'id': '00000000-0000-0000-0000-000000000000', 'status': 'active', 'checksum': 'abcdefghijklmnopqrstuvwxyz012345', 'os_hash_algo': 'supersecure', 'os_hash_value': 'a' * 32 + 'b' * 32 + 'c' * 32 + 'd' * 32, 'size': 9001, 'virtual_size': 9001, 'created_at': ISOTIME, 'updated_at': ISOTIME}
        for key, value in samples.items():
            request = self._get_fake_patch_request()
            body = [{'op': 'replace', 'path': '/%s' % key, 'value': value}]
            request.body = jsonutils.dump_as_bytes(body)
            try:
                self.deserializer.update(request)
            except webob.exc.HTTPForbidden:
                pass
            else:
                self.fail('Updating %s did not result in HTTPForbidden' % key)

    def test_update_reserved_attributes(self):
        samples = {'deleted': False, 'deleted_at': ISOTIME, 'os_glance_import_task': 'foo', 'os_glance_anything': 'bar', 'os_glance_': 'baz', 'os_glance': 'bat'}
        for key, value in samples.items():
            request = self._get_fake_patch_request()
            body = [{'op': 'replace', 'path': '/%s' % key, 'value': value}]
            request.body = jsonutils.dump_as_bytes(body)
            try:
                self.deserializer.update(request)
            except webob.exc.HTTPForbidden:
                pass
            else:
                self.fail('Updating %s did not result in HTTPForbidden' % key)

    def test_update_invalid_attributes(self):
        keys = ['noslash', '///twoslash', '/two/   /slash', '/      /      ', '/trailingslash/', '/lone~tilde', '/trailingtilde~']
        for key in keys:
            request = self._get_fake_patch_request()
            body = [{'op': 'replace', 'path': '%s' % key, 'value': 'dummy'}]
            request.body = jsonutils.dump_as_bytes(body)
            try:
                self.deserializer.update(request)
            except webob.exc.HTTPBadRequest:
                pass
            else:
                self.fail('Updating %s did not result in HTTPBadRequest' % key)

    def test_update_pointer_encoding(self):
        samples = {'/keywith~1slash': ['keywith/slash'], '/keywith~0tilde': ['keywith~tilde'], '/tricky~01': ['tricky~1']}
        for encoded, decoded in samples.items():
            request = self._get_fake_patch_request()
            doc = [{'op': 'replace', 'path': '%s' % encoded, 'value': 'dummy'}]
            request.body = jsonutils.dump_as_bytes(doc)
            output = self.deserializer.update(request)
            self.assertEqual(decoded, output['changes'][0]['path'])

    def test_update_deep_limited_attributes(self):
        samples = {'locations/1/2': []}
        for key, value in samples.items():
            request = self._get_fake_patch_request()
            body = [{'op': 'replace', 'path': '/%s' % key, 'value': value}]
            request.body = jsonutils.dump_as_bytes(body)
            try:
                self.deserializer.update(request)
            except webob.exc.HTTPBadRequest:
                pass
            else:
                self.fail('Updating %s did not result in HTTPBadRequest' % key)

    def test_update_v2_1_missing_operations(self):
        request = self._get_fake_patch_request()
        body = [{'path': '/colburn', 'value': 'arcata'}]
        request.body = jsonutils.dump_as_bytes(body)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.update, request)

    def test_update_v2_1_missing_value(self):
        request = self._get_fake_patch_request()
        body = [{'op': 'replace', 'path': '/colburn'}]
        request.body = jsonutils.dump_as_bytes(body)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.update, request)

    def test_update_v2_1_missing_path(self):
        request = self._get_fake_patch_request()
        body = [{'op': 'replace', 'value': 'arcata'}]
        request.body = jsonutils.dump_as_bytes(body)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.update, request)

    def test_update_v2_0_multiple_operations(self):
        request = self._get_fake_patch_request(content_type_minor_version=0)
        body = [{'replace': '/foo', 'add': '/bar', 'value': 'snore'}]
        request.body = jsonutils.dump_as_bytes(body)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.update, request)

    def test_update_v2_0_missing_operations(self):
        request = self._get_fake_patch_request(content_type_minor_version=0)
        body = [{'value': 'arcata'}]
        request.body = jsonutils.dump_as_bytes(body)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.update, request)

    def test_update_v2_0_missing_value(self):
        request = self._get_fake_patch_request(content_type_minor_version=0)
        body = [{'replace': '/colburn'}]
        request.body = jsonutils.dump_as_bytes(body)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.update, request)

    def test_index(self):
        marker = str(uuid.uuid4())
        path = '/images?limit=1&marker=%s&member_status=pending' % marker
        request = unit_test_utils.get_fake_request(path)
        expected = {'limit': 1, 'marker': marker, 'sort_key': ['created_at'], 'sort_dir': ['desc'], 'member_status': 'pending', 'filters': {}}
        output = self.deserializer.index(request)
        self.assertEqual(expected, output)

    def test_index_with_filter(self):
        name = 'My Little Image'
        path = '/images?name=%s' % name
        request = unit_test_utils.get_fake_request(path)
        output = self.deserializer.index(request)
        self.assertEqual(name, output['filters']['name'])

    def test_index_strip_params_from_filters(self):
        name = 'My Little Image'
        path = '/images?name=%s' % name
        request = unit_test_utils.get_fake_request(path)
        output = self.deserializer.index(request)
        self.assertEqual(name, output['filters']['name'])
        self.assertEqual(1, len(output['filters']))

    def test_index_with_many_filter(self):
        name = 'My Little Image'
        instance_id = str(uuid.uuid4())
        path = '/images?name=%(name)s&id=%(instance_id)s' % {'name': name, 'instance_id': instance_id}
        request = unit_test_utils.get_fake_request(path)
        output = self.deserializer.index(request)
        self.assertEqual(name, output['filters']['name'])
        self.assertEqual(instance_id, output['filters']['id'])

    def test_index_with_filter_and_limit(self):
        name = 'My Little Image'
        path = '/images?name=%s&limit=1' % name
        request = unit_test_utils.get_fake_request(path)
        output = self.deserializer.index(request)
        self.assertEqual(name, output['filters']['name'])
        self.assertEqual(1, output['limit'])

    def test_index_non_integer_limit(self):
        request = unit_test_utils.get_fake_request('/images?limit=blah')
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)

    def test_index_zero_limit(self):
        request = unit_test_utils.get_fake_request('/images?limit=0')
        expected = {'limit': 0, 'sort_key': ['created_at'], 'member_status': 'accepted', 'sort_dir': ['desc'], 'filters': {}}
        output = self.deserializer.index(request)
        self.assertEqual(expected, output)

    def test_index_negative_limit(self):
        request = unit_test_utils.get_fake_request('/images?limit=-1')
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)

    def test_index_fraction(self):
        request = unit_test_utils.get_fake_request('/images?limit=1.1')
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)

    def test_index_invalid_status(self):
        path = '/images?member_status=blah'
        request = unit_test_utils.get_fake_request(path)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)

    def test_index_marker(self):
        marker = str(uuid.uuid4())
        path = '/images?marker=%s' % marker
        request = unit_test_utils.get_fake_request(path)
        output = self.deserializer.index(request)
        self.assertEqual(marker, output.get('marker'))

    def test_index_marker_not_specified(self):
        request = unit_test_utils.get_fake_request('/images')
        output = self.deserializer.index(request)
        self.assertNotIn('marker', output)

    def test_index_limit_not_specified(self):
        request = unit_test_utils.get_fake_request('/images')
        output = self.deserializer.index(request)
        self.assertNotIn('limit', output)

    def test_index_sort_key_id(self):
        request = unit_test_utils.get_fake_request('/images?sort_key=id')
        output = self.deserializer.index(request)
        expected = {'sort_key': ['id'], 'sort_dir': ['desc'], 'member_status': 'accepted', 'filters': {}}
        self.assertEqual(expected, output)

    def test_index_multiple_sort_keys(self):
        request = unit_test_utils.get_fake_request('/images?sort_key=name&sort_key=size')
        output = self.deserializer.index(request)
        expected = {'sort_key': ['name', 'size'], 'sort_dir': ['desc'], 'member_status': 'accepted', 'filters': {}}
        self.assertEqual(expected, output)

    def test_index_invalid_multiple_sort_keys(self):
        request = unit_test_utils.get_fake_request('/images?sort_key=name&sort_key=blah')
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)

    def test_index_sort_dir_asc(self):
        request = unit_test_utils.get_fake_request('/images?sort_dir=asc')
        output = self.deserializer.index(request)
        expected = {'sort_key': ['created_at'], 'sort_dir': ['asc'], 'member_status': 'accepted', 'filters': {}}
        self.assertEqual(expected, output)

    def test_index_multiple_sort_dirs(self):
        req_string = '/images?sort_key=name&sort_dir=asc&sort_key=id&sort_dir=desc'
        request = unit_test_utils.get_fake_request(req_string)
        output = self.deserializer.index(request)
        expected = {'sort_key': ['name', 'id'], 'sort_dir': ['asc', 'desc'], 'member_status': 'accepted', 'filters': {}}
        self.assertEqual(expected, output)

    def test_index_new_sorting_syntax_single_key_default_dir(self):
        req_string = '/images?sort=name'
        request = unit_test_utils.get_fake_request(req_string)
        output = self.deserializer.index(request)
        expected = {'sort_key': ['name'], 'sort_dir': ['desc'], 'member_status': 'accepted', 'filters': {}}
        self.assertEqual(expected, output)

    def test_index_new_sorting_syntax_single_key_desc_dir(self):
        req_string = '/images?sort=name:desc'
        request = unit_test_utils.get_fake_request(req_string)
        output = self.deserializer.index(request)
        expected = {'sort_key': ['name'], 'sort_dir': ['desc'], 'member_status': 'accepted', 'filters': {}}
        self.assertEqual(expected, output)

    def test_index_new_sorting_syntax_multiple_keys_default_dir(self):
        req_string = '/images?sort=name,size'
        request = unit_test_utils.get_fake_request(req_string)
        output = self.deserializer.index(request)
        expected = {'sort_key': ['name', 'size'], 'sort_dir': ['desc', 'desc'], 'member_status': 'accepted', 'filters': {}}
        self.assertEqual(expected, output)

    def test_index_new_sorting_syntax_multiple_keys_asc_dir(self):
        req_string = '/images?sort=name:asc,size:asc'
        request = unit_test_utils.get_fake_request(req_string)
        output = self.deserializer.index(request)
        expected = {'sort_key': ['name', 'size'], 'sort_dir': ['asc', 'asc'], 'member_status': 'accepted', 'filters': {}}
        self.assertEqual(expected, output)

    def test_index_new_sorting_syntax_multiple_keys_different_dirs(self):
        req_string = '/images?sort=name:desc,size:asc'
        request = unit_test_utils.get_fake_request(req_string)
        output = self.deserializer.index(request)
        expected = {'sort_key': ['name', 'size'], 'sort_dir': ['desc', 'asc'], 'member_status': 'accepted', 'filters': {}}
        self.assertEqual(expected, output)

    def test_index_new_sorting_syntax_multiple_keys_optional_dir(self):
        req_string = '/images?sort=name:asc,size'
        request = unit_test_utils.get_fake_request(req_string)
        output = self.deserializer.index(request)
        expected = {'sort_key': ['name', 'size'], 'sort_dir': ['asc', 'desc'], 'member_status': 'accepted', 'filters': {}}
        self.assertEqual(expected, output)
        req_string = '/images?sort=name,size:asc'
        request = unit_test_utils.get_fake_request(req_string)
        output = self.deserializer.index(request)
        expected = {'sort_key': ['name', 'size'], 'sort_dir': ['desc', 'asc'], 'member_status': 'accepted', 'filters': {}}
        self.assertEqual(expected, output)
        req_string = '/images?sort=name,id:asc,size'
        request = unit_test_utils.get_fake_request(req_string)
        output = self.deserializer.index(request)
        expected = {'sort_key': ['name', 'id', 'size'], 'sort_dir': ['desc', 'asc', 'desc'], 'member_status': 'accepted', 'filters': {}}
        self.assertEqual(expected, output)
        req_string = '/images?sort=name:asc,id,size:asc'
        request = unit_test_utils.get_fake_request(req_string)
        output = self.deserializer.index(request)
        expected = {'sort_key': ['name', 'id', 'size'], 'sort_dir': ['asc', 'desc', 'asc'], 'member_status': 'accepted', 'filters': {}}
        self.assertEqual(expected, output)

    def test_index_sort_wrong_sort_dirs_number(self):
        req_string = '/images?sort_key=name&sort_dir=asc&sort_dir=desc'
        request = unit_test_utils.get_fake_request(req_string)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)

    def test_index_sort_dirs_fewer_than_keys(self):
        req_string = '/images?sort_key=name&sort_dir=asc&sort_key=id&sort_dir=asc&sort_key=created_at'
        request = unit_test_utils.get_fake_request(req_string)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)

    def test_index_sort_wrong_sort_dirs_number_without_key(self):
        req_string = '/images?sort_dir=asc&sort_dir=desc'
        request = unit_test_utils.get_fake_request(req_string)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)

    def test_index_sort_private_key(self):
        request = unit_test_utils.get_fake_request('/images?sort_key=min_ram')
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)

    def test_index_sort_key_invalid_value(self):
        request = unit_test_utils.get_fake_request('/images?sort_key=blah')
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)

    def test_index_sort_dir_invalid_value(self):
        request = unit_test_utils.get_fake_request('/images?sort_dir=foo')
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)

    def test_index_new_sorting_syntax_invalid_request(self):
        req_string = '/images?sort=blah'
        request = unit_test_utils.get_fake_request(req_string)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)
        req_string = '/images?sort=name,blah'
        request = unit_test_utils.get_fake_request(req_string)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)
        req_string = '/images?sort=name:foo'
        request = unit_test_utils.get_fake_request(req_string)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)
        req_string = '/images?sort=name:asc:desc'
        request = unit_test_utils.get_fake_request(req_string)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)

    def test_index_combined_sorting_syntax(self):
        req_string = '/images?sort_dir=name&sort=name'
        request = unit_test_utils.get_fake_request(req_string)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)

    def test_index_with_tag(self):
        path = '/images?tag=%s&tag=%s' % ('x86', '64bit')
        request = unit_test_utils.get_fake_request(path)
        output = self.deserializer.index(request)
        self.assertEqual(sorted(['x86', '64bit']), sorted(output['filters']['tags']))

    def test_image_import(self):
        self.config(enabled_import_methods=['party-time'])
        request = unit_test_utils.get_fake_request()
        import_body = {'method': {'name': 'party-time'}}
        request.body = jsonutils.dump_as_bytes(import_body)
        output = self.deserializer.import_image(request)
        expected = {'body': import_body}
        self.assertEqual(expected, output)

    def test_import_image_invalid_body(self):
        request = unit_test_utils.get_fake_request()
        import_body = {'method1': {'name': 'glance-direct'}}
        request.body = jsonutils.dump_as_bytes(import_body)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.import_image, request)

    def test_import_image_invalid_input(self):
        request = unit_test_utils.get_fake_request()
        import_body = {'method': {'abcd': 'glance-direct'}}
        request.body = jsonutils.dump_as_bytes(import_body)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.import_image, request)

    def test_import_image_with_all_stores_not_boolean(self):
        request = unit_test_utils.get_fake_request()
        import_body = {'method': {'name': 'glance-direct'}, 'all_stores': 'true'}
        request.body = jsonutils.dump_as_bytes(import_body)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.import_image, request)

    def test_import_image_with_allow_failure_not_boolean(self):
        request = unit_test_utils.get_fake_request()
        import_body = {'method': {'name': 'glance-direct'}, 'all_stores_must_succeed': 'true'}
        request.body = jsonutils.dump_as_bytes(import_body)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.import_image, request)

    def _get_request_for_method(self, method_name):
        request = unit_test_utils.get_fake_request()
        import_body = {'method': {'name': method_name}}
        request.body = jsonutils.dump_as_bytes(import_body)
        return request
    KNOWN_IMPORT_METHODS = ['glance-direct', 'web-download', 'glance-download']

    def test_import_image_invalid_import_method(self):
        self.config(enabled_import_methods=['bad-method-name'])
        for m in self.KNOWN_IMPORT_METHODS:
            request = self._get_request_for_method(m)
            self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.import_image, request)