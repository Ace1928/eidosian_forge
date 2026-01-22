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
class TestMultiImagesController(base.MultiIsolatedUnitTest):

    def setUp(self):
        super(TestMultiImagesController, self).setUp()
        self.db = unit_test_utils.FakeDB(initialize=False)
        self.policy = unit_test_utils.FakePolicyEnforcer()
        self.notifier = unit_test_utils.FakeNotifier()
        self.store = store
        self._create_images()
        self._create_image_members()
        stores = {'cheap': 'file', 'fast': 'file', 'empty': 'file'}
        self.config(enabled_backends=stores)
        self.store.register_store_opts(CONF)
        self.controller = glance.api.v2.images.ImagesController(self.db, self.policy, self.notifier, self.store)

    def _create_images(self):
        self.images = [_db_fixture(UUID1, owner=TENANT1, checksum=CHKSUM, name='1', size=256, virtual_size=1024, visibility='public', locations=[{'url': '%s/%s' % (BASE_URI, UUID1), 'metadata': {}, 'status': 'active'}], disk_format='raw', container_format='bare', status='active', created_at=DATETIME), _db_fixture(UUID2, owner=TENANT1, checksum=CHKSUM1, name='2', size=512, virtual_size=2048, visibility='public', disk_format='raw', container_format='bare', status='active', tags=['redhat', '64bit', 'power'], properties={'hypervisor_type': 'kvm', 'foo': 'bar', 'bar': 'foo'}, locations=[{'url': 'file://%s/%s' % (self.test_dir, UUID2), 'metadata': {}, 'status': 'active'}], created_at=DATETIME + datetime.timedelta(seconds=1)), _db_fixture(UUID5, owner=TENANT3, checksum=CHKSUM1, name='2', size=512, virtual_size=2048, visibility='public', disk_format='raw', container_format='bare', status='active', tags=['redhat', '64bit', 'power'], properties={'hypervisor_type': 'kvm', 'foo': 'bar', 'bar': 'foo'}, locations=[{'url': 'file://%s/%s' % (self.test_dir, UUID2), 'metadata': {}, 'status': 'active'}], created_at=DATETIME + datetime.timedelta(seconds=1)), _db_fixture(UUID3, owner=TENANT3, checksum=CHKSUM1, name='3', size=512, virtual_size=2048, visibility='public', tags=['windows', '64bit', 'x86'], created_at=DATETIME + datetime.timedelta(seconds=2)), _db_fixture(UUID4, owner=TENANT4, name='4', size=1024, virtual_size=3072, created_at=DATETIME + datetime.timedelta(seconds=3)), _db_fixture(UUID6, owner=TENANT3, checksum=CHKSUM1, name='3', size=512, virtual_size=2048, visibility='public', disk_format='raw', container_format='bare', status='active', tags=['redhat', '64bit', 'power'], properties={'hypervisor_type': 'kvm', 'foo': 'bar', 'bar': 'foo'}, locations=[{'url': 'file://%s/%s' % (self.test_dir, UUID6), 'metadata': {'store': 'fast'}, 'status': 'active'}, {'url': 'file://%s/%s' % (self.test_dir2, UUID6), 'metadata': {'store': 'cheap'}, 'status': 'active'}], created_at=DATETIME + datetime.timedelta(seconds=1)), _db_fixture(UUID7, owner=TENANT3, checksum=CHKSUM1, name='3', size=512, virtual_size=2048, visibility='public', disk_format='raw', container_format='bare', status='active', tags=['redhat', '64bit', 'power'], properties={'hypervisor_type': 'kvm', 'foo': 'bar', 'bar': 'foo'}, locations=[{'url': 'file://%s/%s' % (self.test_dir, UUID7), 'metadata': {'store': 'fast'}, 'status': 'active'}], created_at=DATETIME + datetime.timedelta(seconds=1))]
        [self.db.image_create(None, image) for image in self.images]
        self.db.image_tag_set_all(None, UUID1, ['ping', 'pong'])

    def _create_image_members(self):
        self.image_members = [_db_image_member_fixture(UUID4, TENANT2), _db_image_member_fixture(UUID4, TENANT3, status='accepted')]
        [self.db.image_member_create(None, image_member) for image_member in self.image_members]

    def test_image_import_image_not_exist(self):
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.import_image, request, 'invalid_image', {'method': {'name': 'glance-direct'}})

    def test_image_import_with_active_image(self):
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPConflict, self.controller.import_image, request, UUID2, {'method': {'name': 'glance-direct'}})

    def test_delete_from_store_as_non_owner(self):
        enforcer = unit_test_utils.enforcer_from_rules({'get_image': '', 'delete_image_location': "'TENANT4':%(owner)s", 'get_image_location': ''})
        request = unit_test_utils.get_fake_request()
        self.controller.policy = enforcer
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.delete_from_store, request, 'fast', UUID6)

    def test_delete_from_store_non_active(self):
        request = unit_test_utils.get_fake_request(tenant=TENANT3)
        self.assertRaises(webob.exc.HTTPConflict, self.controller.delete_from_store, request, 'fast', UUID3)

    def test_delete_from_store_no_image(self):
        request = unit_test_utils.get_fake_request(tenant=TENANT3)
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete_from_store, request, 'fast', 'nonexisting')

    def test_delete_from_store_invalid_store(self):
        request = unit_test_utils.get_fake_request(tenant=TENANT3)
        self.assertRaises(webob.exc.HTTPConflict, self.controller.delete_from_store, request, 'burn', UUID6)

    def test_delete_from_store_not_in_store(self):
        request = unit_test_utils.get_fake_request(tenant=TENANT3)
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete_from_store, request, 'empty', UUID6)

    def test_delete_from_store_one_location(self):
        request = unit_test_utils.get_fake_request(tenant=TENANT3)
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.delete_from_store, request, 'fast', UUID7)

    def test_delete_from_store_as_non_admin(self):
        request = unit_test_utils.get_fake_request(tenant=TENANT3)
        self.controller.delete_from_store(request, 'fast', UUID6)
        image = self.controller.show(request, UUID6)
        self.assertEqual(1, len(image.locations))
        self.assertEqual('cheap', image.locations[0]['metadata']['store'])

    def test_delete_from_store_as_admin(self):
        request = unit_test_utils.get_fake_request(is_admin=True)
        self.controller.delete_from_store(request, 'fast', UUID6)
        image = self.controller.show(request, UUID6)
        self.assertEqual(1, len(image.locations))
        self.assertEqual('cheap', image.locations[0]['metadata']['store'])

    def test_image_lazy_loading_store(self):
        existing_image = self.images[1]
        self.assertNotIn('store', existing_image['locations'][0]['metadata'])
        request = unit_test_utils.get_fake_request()
        with mock.patch.object(store_utils, '_get_store_id_from_uri') as mock_uri:
            mock_uri.return_value = 'fast'
            image = self.controller.show(request, UUID2)
            for loc in image.locations:
                self.assertIn('store', loc['metadata'])

    def test_image_lazy_loading_store_different_owner(self):
        existing_image = self.images[2]
        self.assertNotIn('store', existing_image['locations'][0]['metadata'])
        request = unit_test_utils.get_fake_request()
        request.headers.update({'X-Tenant_id': TENANT1})
        with mock.patch.object(store_utils, '_get_store_id_from_uri') as mock_uri:
            mock_uri.return_value = 'fast'
            image = self.controller.show(request, UUID5)
            for loc in image.locations:
                self.assertIn('store', loc['metadata'])

    def test_image_import_invalid_backend_in_request_header(self):
        request = unit_test_utils.get_fake_request()
        request.headers['x-image-meta-store'] = 'dummy'
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage(status='uploading')
            self.assertRaises(webob.exc.HTTPConflict, self.controller.import_image, request, UUID4, {'method': {'name': 'glance-direct'}})

    def test_image_import_raises_conflict_if_disk_format_is_none(self):
        request = unit_test_utils.get_fake_request()
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage(disk_format=None)
            self.assertRaises(webob.exc.HTTPConflict, self.controller.import_image, request, UUID4, {'method': {'name': 'glance-direct'}})

    def test_image_import_raises_conflict(self):
        request = unit_test_utils.get_fake_request()
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage(status='queued')
            self.assertRaises(webob.exc.HTTPConflict, self.controller.import_image, request, UUID4, {'method': {'name': 'glance-direct'}})

    def test_image_import_raises_conflict_for_web_download(self):
        request = unit_test_utils.get_fake_request()
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage()
            self.assertRaises(webob.exc.HTTPConflict, self.controller.import_image, request, UUID4, {'method': {'name': 'web-download'}})

    def test_copy_image_stores_specified_in_header_and_body(self):
        request = unit_test_utils.get_fake_request()
        request.headers['x-image-meta-store'] = 'fast'
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage()
            self.assertRaises(webob.exc.HTTPBadRequest, self.controller.import_image, request, UUID7, {'method': {'name': 'copy-image'}, 'stores': ['fast']})

    def test_copy_image_non_existing_image(self):
        request = unit_test_utils.get_fake_request()
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.side_effect = exception.NotFound
            self.assertRaises(webob.exc.HTTPNotFound, self.controller.import_image, request, UUID1, {'method': {'name': 'copy-image'}, 'stores': ['fast']})

    def test_copy_image_with_all_stores(self):
        request = unit_test_utils.get_fake_request()
        locations = ({'url': 'file://%s/%s' % (self.test_dir, UUID7), 'metadata': {'store': 'fast'}, 'status': 'active'},)
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            with mock.patch.object(self.store, 'get_store_from_store_identifier'):
                mock_get.return_value = FakeImage(id=UUID7, status='active', locations=locations)
                self.assertIsNotNone(self.controller.import_image(request, UUID7, {'method': {'name': 'copy-image'}, 'all_stores': True}))

    def test_copy_non_active_image(self):
        request = unit_test_utils.get_fake_request()
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage(status='uploading')
            self.assertRaises(webob.exc.HTTPConflict, self.controller.import_image, request, UUID1, {'method': {'name': 'copy-image'}, 'stores': ['fast']})

    def test_copy_image_in_existing_store(self):
        request = unit_test_utils.get_fake_request(tenant=TENANT3)
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.import_image, request, UUID6, {'method': {'name': 'copy-image'}, 'stores': ['fast']})

    def test_copy_image_to_other_stores(self):
        request = unit_test_utils.get_fake_request()
        locations = ({'url': 'file://%s/%s' % (self.test_dir, UUID7), 'metadata': {'store': 'fast'}, 'status': 'active'},)
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage(id=UUID7, status='active', locations=locations)
            output = self.controller.import_image(request, UUID7, {'method': {'name': 'copy-image'}, 'stores': ['cheap']})
        self.assertEqual(UUID7, output)