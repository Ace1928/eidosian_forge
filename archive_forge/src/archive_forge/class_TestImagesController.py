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
class TestImagesController(base.IsolatedUnitTest):

    def setUp(self):
        super(TestImagesController, self).setUp()
        self.db = unit_test_utils.FakeDB(initialize=False)
        self.policy = unit_test_utils.FakePolicyEnforcer()
        self.notifier = unit_test_utils.FakeNotifier()
        self.store = unit_test_utils.FakeStoreAPI()
        for i in range(1, 4):
            self.store.data['%s/fake_location_%i' % (BASE_URI, i)] = ('Z', 1)
        self.store_utils = unit_test_utils.FakeStoreUtils(self.store)
        self._create_images()
        self._create_image_members()
        self.controller = glance.api.v2.images.ImagesController(self.db, self.policy, self.notifier, self.store)
        self.action_controller = glance.api.v2.image_actions.ImageActionsController(self.db, self.policy, self.notifier, self.store)
        self.controller.gateway.store_utils = self.store_utils
        self.controller._key_manager = fake_keymgr.fake_api()
        store.create_stores()

    def _create_images(self):
        self.images = [_db_fixture(UUID1, owner=TENANT1, checksum=CHKSUM, os_hash_algo=FAKEHASHALGO, os_hash_value=MULTIHASH1, name='1', size=256, virtual_size=1024, visibility='public', locations=[{'url': '%s/%s' % (BASE_URI, UUID1), 'metadata': {}, 'status': 'active'}], disk_format='raw', container_format='bare', status='active', created_at=DATETIME, updated_at=DATETIME), _db_fixture(UUID2, owner=TENANT1, checksum=CHKSUM1, os_hash_algo=FAKEHASHALGO, os_hash_value=MULTIHASH2, name='2', size=512, virtual_size=2048, visibility='public', disk_format='raw', container_format='bare', status='active', tags=['redhat', '64bit', 'power'], properties={'hypervisor_type': 'kvm', 'foo': 'bar', 'bar': 'foo'}, created_at=DATETIME + datetime.timedelta(seconds=1), updated_at=DATETIME + datetime.timedelta(seconds=1)), _db_fixture(UUID3, owner=TENANT3, checksum=CHKSUM1, os_hash_algo=FAKEHASHALGO, os_hash_value=MULTIHASH2, name='3', size=512, virtual_size=2048, visibility='public', tags=['windows', '64bit', 'x86'], created_at=DATETIME + datetime.timedelta(seconds=2), updated_at=DATETIME + datetime.timedelta(seconds=2)), _db_fixture(UUID4, owner=TENANT4, name='4', size=1024, virtual_size=3072, created_at=DATETIME + datetime.timedelta(seconds=3), updated_at=DATETIME + datetime.timedelta(seconds=3))]
        [self.db.image_create(None, image) for image in self.images]
        self.tasks = [_db_task_fixtures(TASK_ID_1, image_id=UUID1, status='completed', input={'image_id': UUID1, 'import_req': {'method': {'name': 'glance-direct'}, 'backend': ['fake-store']}}, user_id='fake-user-id', request_id='fake-request-id'), _db_task_fixtures(TASK_ID_2, image_id=UUID1, status='completed', input={'image_id': UUID1, 'import_req': {'method': {'name': 'copy-image'}, 'all_stores': True, 'all_stores_must_succeed': False, 'backend': ['fake-store', 'fake_store_1']}}, user_id='fake-user-id', request_id='fake-request-id'), _db_task_fixtures(TASK_ID_3, status='completed', input={'image_id': UUID2, 'import_req': {'method': {'name': 'glance-direct'}, 'backend': ['fake-store']}})]
        [self.db.task_create(None, task) for task in self.tasks]
        self.db.image_tag_set_all(None, UUID1, ['ping', 'pong'])

    def _create_image_members(self):
        self.image_members = [_db_image_member_fixture(UUID4, TENANT2), _db_image_member_fixture(UUID4, TENANT3, status='accepted')]
        [self.db.image_member_create(None, image_member) for image_member in self.image_members]

    def test_index(self):
        self.config(limit_param_default=1, api_limit_max=3)
        request = unit_test_utils.get_fake_request()
        output = self.controller.index(request)
        self.assertEqual(1, len(output['images']))
        actual = set([image.image_id for image in output['images']])
        expected = set([UUID3])
        self.assertEqual(expected, actual)

    def test_index_member_status_accepted(self):
        self.config(limit_param_default=5, api_limit_max=5)
        request = unit_test_utils.get_fake_request(tenant=TENANT2)
        output = self.controller.index(request)
        self.assertEqual(3, len(output['images']))
        actual = set([image.image_id for image in output['images']])
        expected = set([UUID1, UUID2, UUID3])
        self.assertEqual(expected, actual)
        request = unit_test_utils.get_fake_request(tenant=TENANT3)
        output = self.controller.index(request)
        self.assertEqual(4, len(output['images']))
        actual = set([image.image_id for image in output['images']])
        expected = set([UUID1, UUID2, UUID3, UUID4])
        self.assertEqual(expected, actual)

    def test_index_admin(self):
        request = unit_test_utils.get_fake_request(is_admin=True)
        output = self.controller.index(request)
        self.assertEqual(4, len(output['images']))

    def test_index_admin_deleted_images_hidden(self):
        request = unit_test_utils.get_fake_request(is_admin=True)
        self.controller.delete(request, UUID1)
        output = self.controller.index(request)
        self.assertEqual(3, len(output['images']))
        actual = set([image.image_id for image in output['images']])
        expected = set([UUID2, UUID3, UUID4])
        self.assertEqual(expected, actual)

    def test_index_return_parameters(self):
        self.config(limit_param_default=1, api_limit_max=3)
        request = unit_test_utils.get_fake_request()
        output = self.controller.index(request, marker=UUID3, limit=1, sort_key=['created_at'], sort_dir=['desc'])
        self.assertEqual(1, len(output['images']))
        actual = set([image.image_id for image in output['images']])
        expected = set([UUID2])
        self.assertEqual(actual, expected)
        self.assertEqual(UUID2, output['next_marker'])

    def test_index_next_marker(self):
        self.config(limit_param_default=1, api_limit_max=3)
        request = unit_test_utils.get_fake_request()
        output = self.controller.index(request, marker=UUID3, limit=2)
        self.assertEqual(2, len(output['images']))
        actual = set([image.image_id for image in output['images']])
        expected = set([UUID2, UUID1])
        self.assertEqual(expected, actual)
        self.assertEqual(UUID1, output['next_marker'])

    def test_index_no_next_marker(self):
        self.config(limit_param_default=1, api_limit_max=3)
        request = unit_test_utils.get_fake_request()
        output = self.controller.index(request, marker=UUID1, limit=2)
        self.assertEqual(0, len(output['images']))
        actual = set([image.image_id for image in output['images']])
        expected = set([])
        self.assertEqual(expected, actual)
        self.assertNotIn('next_marker', output)

    def test_index_marker_would_be_disallowed(self):
        self.config(limit_param_default=1, api_limit_max=10)
        request = unit_test_utils.get_fake_request(is_admin=True)

        def fake_enforce(context, action, target=None, **kw):
            assert target is not None
            if target['project_id'] != TENANT1:
                raise exception.Forbidden()
        output = self.controller.index(request, sort_dir=['asc'], limit=3)
        self.assertEqual(UUID3, output['next_marker'])
        self.assertEqual(3, len(output['images']))
        with mock.patch.object(self.controller.policy, 'enforce', new=fake_enforce):
            output = self.controller.index(request, sort_dir=['asc'], limit=3)
        self.assertEqual(UUID2, output['next_marker'])
        self.assertEqual(2, len(output['images']))

    def test_index_with_id_filter(self):
        request = unit_test_utils.get_fake_request('/images?id=%s' % UUID1)
        output = self.controller.index(request, filters={'id': UUID1})
        self.assertEqual(1, len(output['images']))
        actual = set([image.image_id for image in output['images']])
        expected = set([UUID1])
        self.assertEqual(expected, actual)

    def test_index_with_invalid_hidden_filter(self):
        request = unit_test_utils.get_fake_request('/images?os_hidden=abcd')
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.index, request, filters={'os_hidden': 'abcd'})

    def test_index_with_checksum_filter_single_image(self):
        req = unit_test_utils.get_fake_request('/images?checksum=%s' % CHKSUM)
        output = self.controller.index(req, filters={'checksum': CHKSUM})
        self.assertEqual(1, len(output['images']))
        actual = list([image.image_id for image in output['images']])
        expected = [UUID1]
        self.assertEqual(expected, actual)

    def test_index_with_checksum_filter_multiple_images(self):
        req = unit_test_utils.get_fake_request('/images?checksum=%s' % CHKSUM1)
        output = self.controller.index(req, filters={'checksum': CHKSUM1})
        self.assertEqual(2, len(output['images']))
        actual = list([image.image_id for image in output['images']])
        expected = [UUID3, UUID2]
        self.assertEqual(expected, actual)

    def test_index_with_non_existent_checksum(self):
        req = unit_test_utils.get_fake_request('/images?checksum=236231827')
        output = self.controller.index(req, filters={'checksum': '236231827'})
        self.assertEqual(0, len(output['images']))

    def test_index_with_os_hash_value_filter_single_image(self):
        req = unit_test_utils.get_fake_request('/images?os_hash_value=%s' % MULTIHASH1)
        output = self.controller.index(req, filters={'os_hash_value': MULTIHASH1})
        self.assertEqual(1, len(output['images']))
        actual = list([image.image_id for image in output['images']])
        expected = [UUID1]
        self.assertEqual(expected, actual)

    def test_index_with_os_hash_value_filter_multiple_images(self):
        req = unit_test_utils.get_fake_request('/images?os_hash_value=%s' % MULTIHASH2)
        output = self.controller.index(req, filters={'os_hash_value': MULTIHASH2})
        self.assertEqual(2, len(output['images']))
        actual = list([image.image_id for image in output['images']])
        expected = [UUID3, UUID2]
        self.assertEqual(expected, actual)

    def test_index_with_non_existent_os_hash_value(self):
        fake_hash_value = hashlib.sha512(b'not_used_in_fixtures').hexdigest()
        req = unit_test_utils.get_fake_request('/images?os_hash_value=%s' % fake_hash_value)
        output = self.controller.index(req, filters={'checksum': fake_hash_value})
        self.assertEqual(0, len(output['images']))

    def test_index_size_max_filter(self):
        request = unit_test_utils.get_fake_request('/images?size_max=512')
        output = self.controller.index(request, filters={'size_max': 512})
        self.assertEqual(3, len(output['images']))
        actual = set([image.image_id for image in output['images']])
        expected = set([UUID1, UUID2, UUID3])
        self.assertEqual(expected, actual)

    def test_index_size_min_filter(self):
        request = unit_test_utils.get_fake_request('/images?size_min=512')
        output = self.controller.index(request, filters={'size_min': 512})
        self.assertEqual(2, len(output['images']))
        actual = set([image.image_id for image in output['images']])
        expected = set([UUID2, UUID3])
        self.assertEqual(expected, actual)

    def test_index_size_range_filter(self):
        path = '/images?size_min=512&size_max=512'
        request = unit_test_utils.get_fake_request(path)
        output = self.controller.index(request, filters={'size_min': 512, 'size_max': 512})
        self.assertEqual(2, len(output['images']))
        actual = set([image.image_id for image in output['images']])
        expected = set([UUID2, UUID3])
        self.assertEqual(expected, actual)

    def test_index_virtual_size_max_filter(self):
        ref = '/images?virtual_size_max=2048'
        request = unit_test_utils.get_fake_request(ref)
        output = self.controller.index(request, filters={'virtual_size_max': 2048})
        self.assertEqual(3, len(output['images']))
        actual = set([image.image_id for image in output['images']])
        expected = set([UUID1, UUID2, UUID3])
        self.assertEqual(expected, actual)

    def test_index_virtual_size_min_filter(self):
        ref = '/images?virtual_size_min=2048'
        request = unit_test_utils.get_fake_request(ref)
        output = self.controller.index(request, filters={'virtual_size_min': 2048})
        self.assertEqual(2, len(output['images']))
        actual = set([image.image_id for image in output['images']])
        expected = set([UUID2, UUID3])
        self.assertEqual(expected, actual)

    def test_index_virtual_size_range_filter(self):
        path = '/images?virtual_size_min=512&virtual_size_max=2048'
        request = unit_test_utils.get_fake_request(path)
        output = self.controller.index(request, filters={'virtual_size_min': 2048, 'virtual_size_max': 2048})
        self.assertEqual(2, len(output['images']))
        actual = set([image.image_id for image in output['images']])
        expected = set([UUID2, UUID3])
        self.assertEqual(expected, actual)

    def test_index_with_invalid_max_range_filter_value(self):
        request = unit_test_utils.get_fake_request('/images?size_max=blah')
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.index, request, filters={'size_max': 'blah'})

    def test_index_with_filters_return_many(self):
        path = '/images?status=queued'
        request = unit_test_utils.get_fake_request(path)
        output = self.controller.index(request, filters={'status': 'queued'})
        self.assertEqual(1, len(output['images']))
        actual = set([image.image_id for image in output['images']])
        expected = set([UUID3])
        self.assertEqual(expected, actual)

    def test_index_with_nonexistent_name_filter(self):
        request = unit_test_utils.get_fake_request('/images?name=%s' % 'blah')
        images = self.controller.index(request, filters={'name': 'blah'})['images']
        self.assertEqual(0, len(images))

    def test_index_with_non_default_is_public_filter(self):
        private_uuid = str(uuid.uuid4())
        new_image = _db_fixture(private_uuid, visibility='private', owner=TENANT3)
        self.db.image_create(None, new_image)
        path = '/images?visibility=private'
        request = unit_test_utils.get_fake_request(path, is_admin=True)
        output = self.controller.index(request, filters={'visibility': 'private'})
        self.assertEqual(1, len(output['images']))
        actual = set([image.image_id for image in output['images']])
        expected = set([private_uuid])
        self.assertEqual(expected, actual)
        path = '/images?visibility=shared'
        request = unit_test_utils.get_fake_request(path, is_admin=True)
        output = self.controller.index(request, filters={'visibility': 'shared'})
        self.assertEqual(1, len(output['images']))
        actual = set([image.image_id for image in output['images']])
        expected = set([UUID4])
        self.assertEqual(expected, actual)

    def test_index_with_many_filters(self):
        url = '/images?status=queued&name=3'
        request = unit_test_utils.get_fake_request(url)
        output = self.controller.index(request, filters={'status': 'queued', 'name': '3'})
        self.assertEqual(1, len(output['images']))
        actual = set([image.image_id for image in output['images']])
        expected = set([UUID3])
        self.assertEqual(expected, actual)

    def test_index_with_marker(self):
        self.config(limit_param_default=1, api_limit_max=3)
        path = '/images'
        request = unit_test_utils.get_fake_request(path)
        output = self.controller.index(request, marker=UUID3)
        actual = set([image.image_id for image in output['images']])
        self.assertEqual(1, len(actual))
        self.assertIn(UUID2, actual)

    def test_index_with_limit(self):
        path = '/images'
        limit = 2
        request = unit_test_utils.get_fake_request(path)
        output = self.controller.index(request, limit=limit)
        actual = set([image.image_id for image in output['images']])
        self.assertEqual(limit, len(actual))
        self.assertIn(UUID3, actual)
        self.assertIn(UUID2, actual)

    def test_index_greater_than_limit_max(self):
        self.config(limit_param_default=1, api_limit_max=3)
        path = '/images'
        request = unit_test_utils.get_fake_request(path)
        output = self.controller.index(request, limit=4)
        actual = set([image.image_id for image in output['images']])
        self.assertEqual(3, len(actual))
        self.assertNotIn(output['next_marker'], output)

    def test_index_default_limit(self):
        self.config(limit_param_default=1, api_limit_max=3)
        path = '/images'
        request = unit_test_utils.get_fake_request(path)
        output = self.controller.index(request)
        actual = set([image.image_id for image in output['images']])
        self.assertEqual(1, len(actual))

    def test_index_with_sort_dir(self):
        path = '/images'
        request = unit_test_utils.get_fake_request(path)
        output = self.controller.index(request, sort_dir=['asc'], limit=3)
        actual = [image.image_id for image in output['images']]
        self.assertEqual(3, len(actual))
        self.assertEqual(UUID1, actual[0])
        self.assertEqual(UUID2, actual[1])
        self.assertEqual(UUID3, actual[2])

    def test_index_with_sort_key(self):
        path = '/images'
        request = unit_test_utils.get_fake_request(path)
        output = self.controller.index(request, sort_key=['created_at'], limit=3)
        actual = [image.image_id for image in output['images']]
        self.assertEqual(3, len(actual))
        self.assertEqual(UUID3, actual[0])
        self.assertEqual(UUID2, actual[1])
        self.assertEqual(UUID1, actual[2])

    def test_index_with_multiple_sort_keys(self):
        path = '/images'
        request = unit_test_utils.get_fake_request(path)
        output = self.controller.index(request, sort_key=['created_at', 'name'], limit=3)
        actual = [image.image_id for image in output['images']]
        self.assertEqual(3, len(actual))
        self.assertEqual(UUID3, actual[0])
        self.assertEqual(UUID2, actual[1])
        self.assertEqual(UUID1, actual[2])

    def test_index_with_marker_not_found(self):
        fake_uuid = str(uuid.uuid4())
        path = '/images'
        request = unit_test_utils.get_fake_request(path)
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.index, request, marker=fake_uuid)

    def test_index_invalid_sort_key(self):
        path = '/images'
        request = unit_test_utils.get_fake_request(path)
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.index, request, sort_key=['foo'])

    def test_index_zero_images(self):
        self.db.reset()
        request = unit_test_utils.get_fake_request()
        output = self.controller.index(request)
        self.assertEqual([], output['images'])

    def test_index_with_tags(self):
        path = '/images?tag=64bit'
        request = unit_test_utils.get_fake_request(path)
        output = self.controller.index(request, filters={'tags': ['64bit']})
        actual = [image.tags for image in output['images']]
        self.assertEqual(2, len(actual))
        self.assertIn('64bit', actual[0])
        self.assertIn('64bit', actual[1])

    def test_index_with_multi_tags(self):
        path = '/images?tag=power&tag=64bit'
        request = unit_test_utils.get_fake_request(path)
        output = self.controller.index(request, filters={'tags': ['power', '64bit']})
        actual = [image.tags for image in output['images']]
        self.assertEqual(1, len(actual))
        self.assertIn('64bit', actual[0])
        self.assertIn('power', actual[0])

    def test_index_with_multi_tags_and_nonexistent(self):
        path = '/images?tag=power&tag=fake'
        request = unit_test_utils.get_fake_request(path)
        output = self.controller.index(request, filters={'tags': ['power', 'fake']})
        actual = [image.tags for image in output['images']]
        self.assertEqual(0, len(actual))

    def test_index_with_tags_and_properties(self):
        path = '/images?tag=64bit&hypervisor_type=kvm'
        request = unit_test_utils.get_fake_request(path)
        output = self.controller.index(request, filters={'tags': ['64bit'], 'hypervisor_type': 'kvm'})
        tags = [image.tags for image in output['images']]
        properties = [image.extra_properties for image in output['images']]
        self.assertEqual(len(tags), len(properties))
        self.assertIn('64bit', tags[0])
        self.assertEqual('kvm', properties[0]['hypervisor_type'])

    def test_index_with_multiple_properties(self):
        path = '/images?foo=bar&hypervisor_type=kvm'
        request = unit_test_utils.get_fake_request(path)
        output = self.controller.index(request, filters={'foo': 'bar', 'hypervisor_type': 'kvm'})
        properties = [image.extra_properties for image in output['images']]
        self.assertEqual('kvm', properties[0]['hypervisor_type'])
        self.assertEqual('bar', properties[0]['foo'])

    def test_index_with_core_and_extra_property(self):
        path = '/images?disk_format=raw&foo=bar'
        request = unit_test_utils.get_fake_request(path)
        output = self.controller.index(request, filters={'foo': 'bar', 'disk_format': 'raw'})
        properties = [image.extra_properties for image in output['images']]
        self.assertEqual(1, len(output['images']))
        self.assertEqual('raw', output['images'][0].disk_format)
        self.assertEqual('bar', properties[0]['foo'])

    def test_index_with_nonexistent_properties(self):
        path = '/images?abc=xyz&pudding=banana'
        request = unit_test_utils.get_fake_request(path)
        output = self.controller.index(request, filters={'abc': 'xyz', 'pudding': 'banana'})
        self.assertEqual(0, len(output['images']))

    def test_index_with_non_existent_tags(self):
        path = '/images?tag=fake'
        request = unit_test_utils.get_fake_request(path)
        output = self.controller.index(request, filters={'tags': ['fake']})
        actual = [image.tags for image in output['images']]
        self.assertEqual(0, len(actual))

    def test_show(self):
        request = unit_test_utils.get_fake_request()
        output = self.controller.show(request, image_id=UUID2)
        self.assertEqual(UUID2, output.image_id)
        self.assertEqual('2', output.name)

    def test_show_deleted_properties(self):
        """Ensure that the api filters out deleted image properties."""
        image = {'id': str(uuid.uuid4()), 'status': 'active', 'properties': {'poo': 'bear'}}
        self.db.image_create(None, image)
        self.db.image_update(None, image['id'], {'properties': {'yin': 'yang'}}, purge_props=True)
        request = unit_test_utils.get_fake_request()
        output = self.controller.show(request, image['id'])
        self.assertEqual('yang', output.extra_properties['yin'])

    def test_show_non_existent(self):
        request = unit_test_utils.get_fake_request()
        image_id = str(uuid.uuid4())
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.show, request, image_id)

    def test_show_deleted_image_admin(self):
        request = unit_test_utils.get_fake_request(is_admin=True)
        self.controller.delete(request, UUID1)
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.show, request, UUID1)

    def test_show_not_allowed(self):
        request = unit_test_utils.get_fake_request()
        self.assertEqual(TENANT1, request.context.project_id)
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.show, request, UUID4)

    def test_show_not_allowed_by_policy(self):
        request = unit_test_utils.get_fake_request(is_admin=True)
        with mock.patch.object(self.controller.policy, 'enforce') as mock_enf:
            mock_enf.side_effect = webob.exc.HTTPForbidden()
            exc = self.assertRaises(webob.exc.HTTPNotFound, self.controller.show, request, UUID4)
        self.assertEqual('The resource could not be found.', str(exc))

    def test_get_task_info(self):
        request = unit_test_utils.get_fake_request()
        output = self.controller.get_task_info(request, image_id=UUID1)
        self.assertEqual(2, len(output['tasks']))
        for task in output['tasks']:
            self.assertEqual(UUID1, task['image_id'])
            self.assertEqual('fake-user-id', task['user_id'])
            self.assertEqual('fake-request-id', task['request_id'])

    def test_get_task_info_no_tasks(self):
        request = unit_test_utils.get_fake_request()
        output = self.controller.get_task_info(request, image_id=UUID2)
        self.assertEqual([], output['tasks'])

    def test_get_task_info_raises_not_found(self):
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.get_task_info, request, 'fake-image-id')

    def test_image_import_raises_conflict_if_container_format_is_none(self):
        request = unit_test_utils.get_fake_request()
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage(container_format=None)
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

    def test_image_import_raises_conflict_for_invalid_status_change(self):
        request = unit_test_utils.get_fake_request()
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage()
            self.assertRaises(webob.exc.HTTPConflict, self.controller.import_image, request, UUID4, {'method': {'name': 'glance-direct'}})

    @mock.patch('glance.db.simple.api.image_set_property_atomic')
    @mock.patch('glance.api.common.get_thread_pool')
    def test_image_import_raises_bad_request(self, mock_gpt, mock_spa):
        request = unit_test_utils.get_fake_request()
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage(status='uploading')
            mock_gpt.return_value.spawn.side_effect = ValueError
            self.assertRaises(webob.exc.HTTPBadRequest, self.controller.import_image, request, UUID4, {'method': {'name': 'glance-direct'}})
            self.assertTrue(mock_gpt.return_value.spawn.called)

    def test_image_import_invalid_uri_filtering(self):
        request = unit_test_utils.get_fake_request()
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage(status='queued')
            self.assertRaises(webob.exc.HTTPBadRequest, self.controller.import_image, request, UUID4, {'method': {'name': 'web-download', 'uri': 'fake_uri'}})

    def test_image_import_raises_bad_request_for_glance_download_missing_input(self):
        request = unit_test_utils.get_fake_request()
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage(status='queued')
            self.assertRaises(webob.exc.HTTPBadRequest, self.controller.import_image, request, UUID4, {'method': {'name': 'glance-download'}})

    def test_image_import_raise_bad_request_wrong_id_for_glance_download(self):
        request = unit_test_utils.get_fake_request()
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage(status='queued')
            self.assertRaises(webob.exc.HTTPBadRequest, self.controller.import_image, request, UUID4, {'method': {'name': 'glance-download', 'glance_image_id': 'fake_id', 'glance_region': 'REGION4'}})

    @mock.patch.object(glance.domain.TaskFactory, 'new_task')
    @mock.patch.object(glance.notifier.ImageRepoProxy, 'get')
    def test_image_import_add_default_service_endpoint_for_glance_download(self, mock_get, mock_nt):
        request = unit_test_utils.get_fake_request()
        mock_get.return_value = FakeImage(status='queued')
        body = {'method': {'name': 'glance-download', 'glance_image_id': UUID4, 'glance_region': 'REGION2'}}
        self.controller.import_image(request, UUID4, body)
        expected_req = {'method': {'name': 'glance-download', 'glance_image_id': UUID4, 'glance_region': 'REGION2', 'glance_service_interface': 'public'}}
        self.assertEqual(expected_req, mock_nt.call_args.kwargs['task_input']['import_req'])

    @mock.patch('glance.context.get_ksa_client')
    def test_image_import_proxies(self, mock_client):
        self.config(worker_self_reference_url='http://glance-worker2.openstack.org')
        request = unit_test_utils.get_fake_request('/v2/images/%s/import' % UUID4)
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage(status='uploading')
            mock_get.return_value.extra_properties['os_glance_stage_host'] = 'https://glance-worker1.openstack.org'
            remote_hdrs = {'x-openstack-request-id': 'remote-req'}
            mock_resp = mock.MagicMock(location='/target', status_code=202, reason='Thanks', headers=remote_hdrs)
            mock_client.return_value.post.return_value = mock_resp
            r = self.controller.import_image(request, UUID4, {'method': {'name': 'glance-direct'}})
            self.assertEqual(UUID4, r)
            mock_client.return_value.post.assert_called_once_with('https://glance-worker1.openstack.org/v2/images/%s/import' % UUID4, json={'method': {'name': 'glance-direct'}}, timeout=60)
            self.assertEqual('remote-req', request.context.request_id)

    @mock.patch('glance.context.get_ksa_client')
    def test_image_delete_proxies(self, mock_client):
        self.config(worker_self_reference_url='http://glance-worker2.openstack.org')
        request = unit_test_utils.get_fake_request('/v2/images/%s' % UUID4, method='DELETE')
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage(status='uploading')
            mock_get.return_value.extra_properties['os_glance_stage_host'] = 'https://glance-worker1.openstack.org'
            remote_hdrs = {'x-openstack-request-id': 'remote-req'}
            mock_resp = mock.MagicMock(location='/target', status_code=202, reason='Thanks', headers=remote_hdrs)
            mock_client.return_value.delete.return_value = mock_resp
            self.controller.delete(request, UUID4)
            mock_client.return_value.delete.assert_called_once_with('https://glance-worker1.openstack.org/v2/images/%s' % UUID4, json=None, timeout=60)

    @mock.patch('glance.context.get_ksa_client')
    def test_image_import_proxies_error(self, mock_client):
        self.config(worker_self_reference_url='http://glance-worker2.openstack.org')
        request = unit_test_utils.get_fake_request('/v2/images/%s/import' % UUID4)
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage(status='uploading')
            mock_get.return_value.extra_properties['os_glance_stage_host'] = 'https://glance-worker1.openstack.org'
            mock_resp = mock.MagicMock(location='/target', status_code=456, reason='No thanks')
            mock_client.return_value.post.return_value = mock_resp
            exc = self.assertRaises(webob.exc.HTTPError, self.controller.import_image, request, UUID4, {'method': {'name': 'glance-direct'}})
            self.assertEqual('456 No thanks', exc.status)
            mock_client.return_value.post.assert_called_once_with('https://glance-worker1.openstack.org/v2/images/%s/import' % UUID4, json={'method': {'name': 'glance-direct'}}, timeout=60)

    @mock.patch('glance.context.get_ksa_client')
    def test_image_delete_proxies_error(self, mock_client):
        self.config(worker_self_reference_url='http://glance-worker2.openstack.org')
        request = unit_test_utils.get_fake_request('/v2/images/%s' % UUID4, method='DELETE')
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage(status='uploading')
            mock_get.return_value.extra_properties['os_glance_stage_host'] = 'https://glance-worker1.openstack.org'
            remote_hdrs = {'x-openstack-request-id': 'remote-req'}
            mock_resp = mock.MagicMock(location='/target', status_code=456, reason='No thanks', headers=remote_hdrs)
            mock_client.return_value.delete.return_value = mock_resp
            exc = self.assertRaises(webob.exc.HTTPError, self.controller.delete, request, UUID4)
            self.assertEqual('456 No thanks', exc.status)
            mock_client.return_value.delete.assert_called_once_with('https://glance-worker1.openstack.org/v2/images/%s' % UUID4, json=None, timeout=60)

    @mock.patch('glance.context.get_ksa_client')
    @mock.patch.object(glance.notifier.ImageRepoProxy, 'get')
    @mock.patch.object(glance.notifier.ImageRepoProxy, 'remove')
    def test_image_delete_deletes_locally_on_error(self, mock_remove, mock_get, mock_client):
        self.config(worker_self_reference_url='http://glance-worker2.openstack.org')
        request = unit_test_utils.get_fake_request('/v2/images/%s' % UUID4, method='DELETE')
        image = FakeImage(status='uploading')
        mock_get.return_value = image
        image.extra_properties['os_glance_stage_host'] = 'https://glance-worker1.openstack.org'
        image.delete = mock.MagicMock()
        mock_client.return_value.delete.side_effect = requests.exceptions.ConnectTimeout
        self.controller.delete(request, UUID4)
        mock_get.return_value.delete.assert_called_once_with()
        mock_remove.assert_called_once_with(image)
        mock_client.return_value.delete.assert_called_once_with('https://glance-worker1.openstack.org/v2/images/%s' % UUID4, json=None, timeout=60)

    @mock.patch('glance.context.get_ksa_client')
    def test_image_import_no_proxy_non_direct(self, mock_client):
        self.config(worker_self_reference_url='http://glance-worker2.openstack.org')
        request = unit_test_utils.get_fake_request('/v2/images/%s/import' % UUID4)
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage(status='queued')
            mock_get.return_value.extra_properties['os_glance_stage_host'] = 'https://glance-worker1.openstack.org'
            self.assertRaises(webob.exc.HTTPBadRequest, self.controller.import_image, request, UUID4, {'method': {'name': 'web-download', 'url': 'not-a-url'}})
            mock_client.return_value.post.assert_not_called()

    def test_create(self):
        request = unit_test_utils.get_fake_request()
        image = {'name': 'image-1'}
        output = self.controller.create(request, image=image, extra_properties={}, tags=[])
        self.assertEqual('image-1', output.name)
        self.assertEqual({}, output.extra_properties)
        self.assertEqual(set([]), output.tags)
        self.assertEqual('shared', output.visibility)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('image.create', output_log['event_type'])
        self.assertEqual('image-1', output_log['payload']['name'])

    def test_create_disabled_notification(self):
        self.config(disabled_notifications=['image.create'])
        request = unit_test_utils.get_fake_request()
        image = {'name': 'image-1'}
        output = self.controller.create(request, image=image, extra_properties={}, tags=[])
        self.assertEqual('image-1', output.name)
        self.assertEqual({}, output.extra_properties)
        self.assertEqual(set([]), output.tags)
        self.assertEqual('shared', output.visibility)
        output_logs = self.notifier.get_logs()
        self.assertEqual(0, len(output_logs))

    def test_create_with_properties(self):
        request = unit_test_utils.get_fake_request()
        image_properties = {'foo': 'bar'}
        image = {'name': 'image-1'}
        output = self.controller.create(request, image=image, extra_properties=image_properties, tags=[])
        self.assertEqual('image-1', output.name)
        self.assertEqual(image_properties, output.extra_properties)
        self.assertEqual(set([]), output.tags)
        self.assertEqual('shared', output.visibility)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('image.create', output_log['event_type'])
        self.assertEqual('image-1', output_log['payload']['name'])

    def test_create_with_too_many_properties(self):
        self.config(image_property_quota=1)
        request = unit_test_utils.get_fake_request()
        image_properties = {'foo': 'bar', 'foo2': 'bar'}
        image = {'name': 'image-1'}
        self.assertRaises(webob.exc.HTTPRequestEntityTooLarge, self.controller.create, request, image=image, extra_properties=image_properties, tags=[])

    def test_create_with_bad_min_disk_size(self):
        request = unit_test_utils.get_fake_request()
        image = {'min_disk': -42, 'name': 'image-1'}
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.create, request, image=image, extra_properties={}, tags=[])

    def test_create_with_bad_min_ram_size(self):
        request = unit_test_utils.get_fake_request()
        image = {'min_ram': -42, 'name': 'image-1'}
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.create, request, image=image, extra_properties={}, tags=[])

    def test_create_public_image_as_admin(self):
        request = unit_test_utils.get_fake_request()
        image = {'name': 'image-1', 'visibility': 'public'}
        output = self.controller.create(request, image=image, extra_properties={}, tags=[])
        self.assertEqual('public', output.visibility)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('image.create', output_log['event_type'])
        self.assertEqual(output.image_id, output_log['payload']['id'])

    def test_create_dup_id(self):
        request = unit_test_utils.get_fake_request()
        image = {'image_id': UUID4}
        self.assertRaises(webob.exc.HTTPConflict, self.controller.create, request, image=image, extra_properties={}, tags=[])

    def test_create_duplicate_tags(self):
        request = unit_test_utils.get_fake_request()
        tags = ['ping', 'ping']
        output = self.controller.create(request, image={}, extra_properties={}, tags=tags)
        self.assertEqual(set(['ping']), output.tags)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('image.create', output_log['event_type'])
        self.assertEqual(output.image_id, output_log['payload']['id'])

    def test_create_with_too_many_tags(self):
        self.config(image_tag_quota=1)
        request = unit_test_utils.get_fake_request()
        tags = ['ping', 'pong']
        self.assertRaises(webob.exc.HTTPRequestEntityTooLarge, self.controller.create, request, image={}, extra_properties={}, tags=tags)

    def test_create_with_owner_non_admin(self):
        enforcer = unit_test_utils.enforcer_from_rules({'add_image': 'role:member,reader'})
        request = unit_test_utils.get_fake_request()
        request.context.is_admin = False
        image = {'owner': '12345'}
        self.controller.policy = enforcer
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.create, request, image=image, extra_properties={}, tags=[])
        enforcer = unit_test_utils.enforcer_from_rules({'add_image': "'{0}':%(owner)s".format(TENANT1)})
        request = unit_test_utils.get_fake_request()
        request.context.is_admin = False
        image = {'owner': TENANT1}
        self.controller.policy = enforcer
        output = self.controller.create(request, image=image, extra_properties={}, tags=[])
        self.assertEqual(TENANT1, output.owner)

    def test_create_with_owner_admin(self):
        request = unit_test_utils.get_fake_request()
        request.context.is_admin = True
        image = {'owner': '12345'}
        output = self.controller.create(request, image=image, extra_properties={}, tags=[])
        self.assertEqual('12345', output.owner)

    def test_create_with_duplicate_location(self):
        request = unit_test_utils.get_fake_request()
        location = {'url': '%s/fake_location' % BASE_URI, 'metadata': {}}
        image = {'name': 'image-1', 'locations': [location, location]}
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.create, request, image=image, extra_properties={}, tags=[])

    def test_create_unexpected_property(self):
        request = unit_test_utils.get_fake_request()
        image_properties = {'unexpected': 'unexpected'}
        image = {'name': 'image-1'}
        with mock.patch.object(domain.ImageFactory, 'new_image', side_effect=TypeError):
            self.assertRaises(webob.exc.HTTPBadRequest, self.controller.create, request, image=image, extra_properties=image_properties, tags=[])

    def test_create_reserved_property(self):
        request = unit_test_utils.get_fake_request()
        image_properties = {'reserved': 'reserved'}
        image = {'name': 'image-1'}
        with mock.patch.object(domain.ImageFactory, 'new_image', side_effect=exception.ReservedProperty(property='reserved')):
            self.assertRaises(webob.exc.HTTPForbidden, self.controller.create, request, image=image, extra_properties=image_properties, tags=[])

    def test_create_readonly_property(self):
        request = unit_test_utils.get_fake_request()
        image_properties = {'readonly': 'readonly'}
        image = {'name': 'image-1'}
        with mock.patch.object(domain.ImageFactory, 'new_image', side_effect=exception.ReadonlyProperty(property='readonly')):
            self.assertRaises(webob.exc.HTTPForbidden, self.controller.create, request, image=image, extra_properties=image_properties, tags=[])

    def test_update_no_changes(self):
        request = unit_test_utils.get_fake_request()
        output = self.controller.update(request, UUID1, changes=[])
        self.assertEqual(UUID1, output.image_id)
        self.assertEqual(output.created_at, output.updated_at)
        self.assertEqual(2, len(output.tags))
        self.assertIn('ping', output.tags)
        self.assertIn('pong', output.tags)
        output_logs = self.notifier.get_logs()
        self.assertEqual(0, len(output_logs))

    def test_update_queued_image_with_hidden(self):
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'replace', 'path': ['os_hidden'], 'value': 'true'}]
        image = self.controller.update(request, UUID1, changes=changes)
        self.assertTrue(image.os_hidden)

    def test_update_with_bad_min_disk(self):
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'replace', 'path': ['min_disk'], 'value': -42}]
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.update, request, UUID1, changes=changes)

    def test_update_with_bad_min_ram(self):
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'replace', 'path': ['min_ram'], 'value': -42}]
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.update, request, UUID1, changes=changes)

    def test_update_image_doesnt_exist(self):
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.update, request, str(uuid.uuid4()), changes=[])

    def test_update_deleted_image_admin(self):
        request = unit_test_utils.get_fake_request(is_admin=True)
        self.controller.delete(request, UUID1)
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.update, request, UUID1, changes=[])

    def test_update_with_too_many_properties(self):
        self.config(show_multiple_locations=True)
        self.config(user_storage_quota='1')
        new_location = {'url': '%s/fake_location' % BASE_URI, 'metadata': {}}
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': new_location}]
        self.assertRaises(webob.exc.HTTPRequestEntityTooLarge, self.controller.update, request, UUID1, changes=changes)

    def test_update_replace_base_attribute(self):
        self.db.image_update(None, UUID1, {'properties': {'foo': 'bar'}})
        request = unit_test_utils.get_fake_request()
        request.context.is_admin = True
        changes = [{'op': 'replace', 'path': ['name'], 'value': 'fedora'}, {'op': 'replace', 'path': ['owner'], 'value': TENANT3}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(UUID1, output.image_id)
        self.assertEqual('fedora', output.name)
        self.assertEqual(TENANT3, output.owner)
        self.assertEqual({'foo': 'bar'}, output.extra_properties)
        self.assertNotEqual(output.created_at, output.updated_at)

    def test_update_replace_onwer_non_admin(self):
        request = unit_test_utils.get_fake_request()
        request.context.is_admin = False
        changes = [{'op': 'replace', 'path': ['owner'], 'value': TENANT3}]
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, UUID1, changes)

    def test_update_replace_tags(self):
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'replace', 'path': ['tags'], 'value': ['king', 'kong']}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(UUID1, output.image_id)
        self.assertEqual(2, len(output.tags))
        self.assertIn('king', output.tags)
        self.assertIn('kong', output.tags)
        self.assertNotEqual(output.created_at, output.updated_at)

    def test_update_replace_property(self):
        request = unit_test_utils.get_fake_request()
        properties = {'foo': 'bar', 'snitch': 'golden'}
        self.db.image_update(None, UUID1, {'properties': properties})
        output = self.controller.show(request, UUID1)
        self.assertEqual('bar', output.extra_properties['foo'])
        self.assertEqual('golden', output.extra_properties['snitch'])
        changes = [{'op': 'replace', 'path': ['foo'], 'value': 'baz'}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(UUID1, output.image_id)
        self.assertEqual('baz', output.extra_properties['foo'])
        self.assertEqual('golden', output.extra_properties['snitch'])
        self.assertNotEqual(output.created_at, output.updated_at)

    def test_update_add_too_many_properties(self):
        self.config(image_property_quota=1)
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['foo'], 'value': 'baz'}, {'op': 'add', 'path': ['snitch'], 'value': 'golden'}]
        self.assertRaises(webob.exc.HTTPRequestEntityTooLarge, self.controller.update, request, UUID1, changes)

    def test_update_reserved_not_counted_in_quota(self):
        self.config(image_property_quota=1)
        request = unit_test_utils.get_fake_request()
        self.db.image_update(None, UUID1, {'properties': {'os_glance_foo': '123', 'os_glance_bar': 456}})
        changes = [{'op': 'add', 'path': ['foo'], 'value': 'baz'}]
        self.controller.update(request, UUID1, changes)
        changes = [{'op': 'add', 'path': ['snitch'], 'value': 'golden'}]
        self.assertRaises(webob.exc.HTTPRequestEntityTooLarge, self.controller.update, request, UUID1, changes)

    def test_update_add_and_remove_too_many_properties(self):
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['foo'], 'value': 'baz'}, {'op': 'add', 'path': ['snitch'], 'value': 'golden'}]
        self.controller.update(request, UUID1, changes)
        self.config(image_property_quota=1)
        changes = [{'op': 'remove', 'path': ['foo']}, {'op': 'add', 'path': ['fizz'], 'value': 'buzz'}]
        self.assertRaises(webob.exc.HTTPRequestEntityTooLarge, self.controller.update, request, UUID1, changes)

    def test_update_add_unlimited_properties(self):
        self.config(image_property_quota=-1)
        request = unit_test_utils.get_fake_request()
        output = self.controller.show(request, UUID1)
        changes = [{'op': 'add', 'path': ['foo'], 'value': 'bar'}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(UUID1, output.image_id)
        self.assertNotEqual(output.created_at, output.updated_at)

    def test_update_format_properties(self):
        statuses_for_immutability = ['active', 'saving', 'killed']
        request = unit_test_utils.get_fake_request(roles=['admin'], is_admin=True)
        for status in statuses_for_immutability:
            image = {'id': str(uuid.uuid4()), 'status': status, 'disk_format': 'ari', 'container_format': 'ari'}
            self.db.image_create(None, image)
            changes = [{'op': 'replace', 'path': ['disk_format'], 'value': 'ami'}]
            self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, image['id'], changes)
            changes = [{'op': 'replace', 'path': ['container_format'], 'value': 'ami'}]
            self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, image['id'], changes)
        self.db.image_update(None, image['id'], {'status': 'queued'})
        changes = [{'op': 'replace', 'path': ['disk_format'], 'value': 'raw'}, {'op': 'replace', 'path': ['container_format'], 'value': 'bare'}]
        resp = self.controller.update(request, image['id'], changes)
        self.assertEqual('raw', resp.disk_format)
        self.assertEqual('bare', resp.container_format)

    def test_update_remove_property_while_over_limit(self):
        """Ensure that image properties can be removed.

        Image properties should be able to be removed as long as the image has
        fewer than the limited number of image properties after the
        transaction.

        """
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['foo'], 'value': 'baz'}, {'op': 'add', 'path': ['snitch'], 'value': 'golden'}, {'op': 'add', 'path': ['fizz'], 'value': 'buzz'}]
        self.controller.update(request, UUID1, changes)
        self.config(image_property_quota=1)
        changes = [{'op': 'remove', 'path': ['foo']}, {'op': 'remove', 'path': ['snitch']}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(UUID1, output.image_id)
        self.assertEqual(1, len(output.extra_properties))
        self.assertEqual('buzz', output.extra_properties['fizz'])
        self.assertNotEqual(output.created_at, output.updated_at)

    def test_update_add_and_remove_property_under_limit(self):
        """Ensure that image properties can be removed.

        Image properties should be able to be added and removed simultaneously
        as long as the image has fewer than the limited number of image
        properties after the transaction.

        """
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['foo'], 'value': 'baz'}, {'op': 'add', 'path': ['snitch'], 'value': 'golden'}]
        self.controller.update(request, UUID1, changes)
        self.config(image_property_quota=1)
        changes = [{'op': 'remove', 'path': ['foo']}, {'op': 'remove', 'path': ['snitch']}, {'op': 'add', 'path': ['fizz'], 'value': 'buzz'}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(UUID1, output.image_id)
        self.assertEqual(1, len(output.extra_properties))
        self.assertEqual('buzz', output.extra_properties['fizz'])
        self.assertNotEqual(output.created_at, output.updated_at)

    def test_update_replace_missing_property(self):
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'replace', 'path': 'foo', 'value': 'baz'}]
        self.assertRaises(webob.exc.HTTPConflict, self.controller.update, request, UUID1, changes)

    def test_prop_protection_with_create_and_permitted_role(self):
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        self.controller = glance.api.v2.images.ImagesController(self.db, enforcer, self.notifier, self.store)
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['admin'])
        image = {'name': 'image-1'}
        created_image = self.controller.create(request, image=image, extra_properties={}, tags=[])
        another_request = unit_test_utils.get_fake_request(roles=['member'])
        changes = [{'op': 'add', 'path': ['x_owner_foo'], 'value': 'bar'}]
        output = self.controller.update(another_request, created_image.image_id, changes)
        self.assertEqual('bar', output.extra_properties['x_owner_foo'])

    def test_prop_protection_with_update_and_permitted_policy(self):
        self.set_property_protections(use_policies=True)
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        self.controller = glance.api.v2.images.ImagesController(self.db, enforcer, self.notifier, self.store)
        request = unit_test_utils.get_fake_request(roles=['spl_role', 'admin'])
        image = {'name': 'image-1'}
        extra_props = {'spl_creator_policy': 'bar'}
        created_image = self.controller.create(request, image=image, extra_properties=extra_props, tags=[])
        self.assertEqual('bar', created_image.extra_properties['spl_creator_policy'])
        another_request = unit_test_utils.get_fake_request(roles=['spl_role'])
        changes = [{'op': 'replace', 'path': ['spl_creator_policy'], 'value': 'par'}]
        enforcer = unit_test_utils.enforcer_from_rules({'get_image': '', 'modify_image': 'role:spl_role'})
        self.controller.policy = enforcer
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, another_request, created_image.image_id, changes)
        enforcer = unit_test_utils.enforcer_from_rules({'get_image': '', 'modify_image': 'role:admin'})
        self.controller.policy = enforcer
        another_request = unit_test_utils.get_fake_request(roles=['admin'])
        output = self.controller.update(another_request, created_image.image_id, changes)
        self.assertEqual('par', output.extra_properties['spl_creator_policy'])

    def test_prop_protection_with_create_with_patch_and_policy(self):
        self.set_property_protections(use_policies=True)
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        self.controller = glance.api.v2.images.ImagesController(self.db, enforcer, self.notifier, self.store)
        request = unit_test_utils.get_fake_request(roles=['spl_role', 'admin'])
        image = {'name': 'image-1'}
        extra_props = {'spl_default_policy': 'bar'}
        created_image = self.controller.create(request, image=image, extra_properties=extra_props, tags=[])
        another_request = unit_test_utils.get_fake_request(roles=['fake_role'])
        changes = [{'op': 'add', 'path': ['spl_creator_policy'], 'value': 'bar'}]
        enforcer = unit_test_utils.enforcer_from_rules({'get_image': '', 'modify_image': 'role:fake_role'})
        self.controller.policy = enforcer
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, another_request, created_image.image_id, changes)
        enforcer = unit_test_utils.enforcer_from_rules({'get_image': '', 'modify_image': 'role:member'})
        self.controller.policy = enforcer
        another_request = unit_test_utils.get_fake_request(roles=['member', 'spl_role'])
        output = self.controller.update(another_request, created_image.image_id, changes)
        self.assertEqual('bar', output.extra_properties['spl_creator_policy'])

    def test_prop_protection_with_create_and_unpermitted_role(self):
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        self.controller = glance.api.v2.images.ImagesController(self.db, enforcer, self.notifier, self.store)
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['admin'])
        image = {'name': 'image-1'}
        created_image = self.controller.create(request, image=image, extra_properties={}, tags=[])
        roles = ['fake_member']
        enforcer = unit_test_utils.enforcer_from_rules({'get_image': '', 'modify_image': 'role:fake_member'})
        self.controller.policy = enforcer
        another_request = unit_test_utils.get_fake_request(roles=roles)
        changes = [{'op': 'add', 'path': ['x_owner_foo'], 'value': 'bar'}]
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, another_request, created_image.image_id, changes)

    def test_prop_protection_with_show_and_permitted_role(self):
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        self.controller = glance.api.v2.images.ImagesController(self.db, enforcer, self.notifier, self.store)
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['admin'])
        image = {'name': 'image-1'}
        extra_props = {'x_owner_foo': 'bar'}
        created_image = self.controller.create(request, image=image, extra_properties=extra_props, tags=[])
        another_request = unit_test_utils.get_fake_request(roles=['reader', 'member'])
        output = self.controller.show(another_request, created_image.image_id)
        self.assertEqual('bar', output.extra_properties['x_owner_foo'])

    def test_prop_protection_with_show_and_unpermitted_role(self):
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        self.controller = glance.api.v2.images.ImagesController(self.db, enforcer, self.notifier, self.store)
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['member'])
        image = {'name': 'image-1'}
        extra_props = {'x_owner_foo': 'bar'}
        created_image = self.controller.create(request, image=image, extra_properties=extra_props, tags=[])
        another_request = unit_test_utils.get_fake_request(roles=['reader', 'fake_role'])
        output = self.controller.show(another_request, created_image.image_id)
        self.assertRaises(KeyError, output.extra_properties.__getitem__, 'x_owner_foo')

    def test_prop_protection_with_update_and_permitted_role(self):
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        self.controller = glance.api.v2.images.ImagesController(self.db, enforcer, self.notifier, self.store)
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['admin'])
        image = {'name': 'image-1'}
        extra_props = {'x_owner_foo': 'bar'}
        created_image = self.controller.create(request, image=image, extra_properties=extra_props, tags=[])
        another_request = unit_test_utils.get_fake_request(roles=['member'])
        changes = [{'op': 'replace', 'path': ['x_owner_foo'], 'value': 'baz'}]
        output = self.controller.update(another_request, created_image.image_id, changes)
        self.assertEqual('baz', output.extra_properties['x_owner_foo'])

    def test_prop_protection_with_update_and_unpermitted_role(self):
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        self.controller = glance.api.v2.images.ImagesController(self.db, enforcer, self.notifier, self.store)
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['admin'])
        image = {'name': 'image-1'}
        extra_props = {'x_owner_foo': 'bar'}
        created_image = self.controller.create(request, image=image, extra_properties=extra_props, tags=[])
        enforcer = unit_test_utils.enforcer_from_rules({'get_image': '', 'modify_image': 'role:fake_role'})
        self.controller.policy = enforcer
        another_request = unit_test_utils.get_fake_request(roles=['fake_role'])
        changes = [{'op': 'replace', 'path': ['x_owner_foo'], 'value': 'baz'}]
        self.assertRaises(webob.exc.HTTPConflict, self.controller.update, another_request, created_image.image_id, changes)

    def test_prop_protection_with_delete_and_permitted_role(self):
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        self.controller = glance.api.v2.images.ImagesController(self.db, enforcer, self.notifier, self.store)
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['admin'])
        image = {'name': 'image-1'}
        extra_props = {'x_owner_foo': 'bar'}
        created_image = self.controller.create(request, image=image, extra_properties=extra_props, tags=[])
        another_request = unit_test_utils.get_fake_request(roles=['member'])
        changes = [{'op': 'remove', 'path': ['x_owner_foo']}]
        output = self.controller.update(another_request, created_image.image_id, changes)
        self.assertRaises(KeyError, output.extra_properties.__getitem__, 'x_owner_foo')

    def test_prop_protection_with_delete_and_unpermitted_role(self):
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        self.controller = glance.api.v2.images.ImagesController(self.db, enforcer, self.notifier, self.store)
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['admin'])
        image = {'name': 'image-1'}
        extra_props = {'x_owner_foo': 'bar'}
        created_image = self.controller.create(request, image=image, extra_properties=extra_props, tags=[])
        enforcer = unit_test_utils.enforcer_from_rules({'get_image': '', 'modify_image': 'role:fake_role'})
        self.controller.policy = enforcer
        another_request = unit_test_utils.get_fake_request(roles=['fake_role'])
        changes = [{'op': 'remove', 'path': ['x_owner_foo']}]
        self.assertRaises(webob.exc.HTTPConflict, self.controller.update, another_request, created_image.image_id, changes)

    def test_create_protected_prop_case_insensitive(self):
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        self.controller = glance.api.v2.images.ImagesController(self.db, enforcer, self.notifier, self.store)
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['admin'])
        image = {'name': 'image-1'}
        created_image = self.controller.create(request, image=image, extra_properties={}, tags=[])
        another_request = unit_test_utils.get_fake_request(roles=['member'])
        changes = [{'op': 'add', 'path': ['x_case_insensitive'], 'value': '1'}]
        output = self.controller.update(another_request, created_image.image_id, changes)
        self.assertEqual('1', output.extra_properties['x_case_insensitive'])

    def test_read_protected_prop_case_insensitive(self):
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        self.controller = glance.api.v2.images.ImagesController(self.db, enforcer, self.notifier, self.store)
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['admin'])
        image = {'name': 'image-1'}
        extra_props = {'x_case_insensitive': '1'}
        created_image = self.controller.create(request, image=image, extra_properties=extra_props, tags=[])
        another_request = unit_test_utils.get_fake_request(roles=['reader', 'member'])
        output = self.controller.show(another_request, created_image.image_id)
        self.assertEqual('1', output.extra_properties['x_case_insensitive'])

    def test_update_protected_prop_case_insensitive(self):
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        self.controller = glance.api.v2.images.ImagesController(self.db, enforcer, self.notifier, self.store)
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['admin'])
        image = {'name': 'image-1'}
        extra_props = {'x_case_insensitive': '1'}
        created_image = self.controller.create(request, image=image, extra_properties=extra_props, tags=[])
        another_request = unit_test_utils.get_fake_request(roles=['member'])
        changes = [{'op': 'replace', 'path': ['x_case_insensitive'], 'value': '2'}]
        output = self.controller.update(another_request, created_image.image_id, changes)
        self.assertEqual('2', output.extra_properties['x_case_insensitive'])

    def test_delete_protected_prop_case_insensitive(self):
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        self.controller = glance.api.v2.images.ImagesController(self.db, enforcer, self.notifier, self.store)
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['admin'])
        image = {'name': 'image-1'}
        extra_props = {'x_case_insensitive': 'bar'}
        created_image = self.controller.create(request, image=image, extra_properties=extra_props, tags=[])
        another_request = unit_test_utils.get_fake_request(roles=['member'])
        changes = [{'op': 'remove', 'path': ['x_case_insensitive']}]
        output = self.controller.update(another_request, created_image.image_id, changes)
        self.assertRaises(KeyError, output.extra_properties.__getitem__, 'x_case_insensitive')

    def test_create_non_protected_prop(self):
        """Property marked with special char @ creatable by an unknown role"""
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['admin'])
        image = {'name': 'image-1'}
        extra_props = {'x_all_permitted_1': '1'}
        created_image = self.controller.create(request, image=image, extra_properties=extra_props, tags=[])
        self.assertEqual('1', created_image.extra_properties['x_all_permitted_1'])
        another_request = unit_test_utils.get_fake_request(roles=['joe_soap'])
        extra_props = {'x_all_permitted_2': '2'}
        created_image = self.controller.create(another_request, image=image, extra_properties=extra_props, tags=[])
        self.assertEqual('2', created_image.extra_properties['x_all_permitted_2'])

    def test_read_non_protected_prop(self):
        """Property marked with special char @ readable by an unknown role"""
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['admin'])
        image = {'name': 'image-1'}
        extra_props = {'x_all_permitted': '1'}
        created_image = self.controller.create(request, image=image, extra_properties=extra_props, tags=[])
        another_request = unit_test_utils.get_fake_request(roles=['joe_soap'])
        output = self.controller.show(another_request, created_image.image_id)
        self.assertEqual('1', output.extra_properties['x_all_permitted'])

    def test_update_non_protected_prop(self):
        """Property marked with special char @ updatable by an unknown role"""
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['admin'])
        image = {'name': 'image-1'}
        extra_props = {'x_all_permitted': 'bar'}
        created_image = self.controller.create(request, image=image, extra_properties=extra_props, tags=[])
        another_request = unit_test_utils.get_fake_request(roles=['member', 'joe_soap'])
        changes = [{'op': 'replace', 'path': ['x_all_permitted'], 'value': 'baz'}]
        output = self.controller.update(another_request, created_image.image_id, changes)
        self.assertEqual('baz', output.extra_properties['x_all_permitted'])

    def test_delete_non_protected_prop(self):
        """Property marked with special char @ deletable by an unknown role"""
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['admin'])
        image = {'name': 'image-1'}
        extra_props = {'x_all_permitted': 'bar'}
        created_image = self.controller.create(request, image=image, extra_properties=extra_props, tags=[])
        another_request = unit_test_utils.get_fake_request(roles=['member'])
        changes = [{'op': 'remove', 'path': ['x_all_permitted']}]
        output = self.controller.update(another_request, created_image.image_id, changes)
        self.assertRaises(KeyError, output.extra_properties.__getitem__, 'x_all_permitted')

    def test_create_locked_down_protected_prop(self):
        """Property marked with special char ! creatable by no one"""
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['admin'])
        image = {'name': 'image-1'}
        created_image = self.controller.create(request, image=image, extra_properties={}, tags=[])
        roles = ['fake_member']
        another_request = unit_test_utils.get_fake_request(roles=roles)
        changes = [{'op': 'add', 'path': ['x_none_permitted'], 'value': 'bar'}]
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, another_request, created_image.image_id, changes)

    def test_read_locked_down_protected_prop(self):
        """Property marked with special char ! readable by no one"""
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['member'])
        image = {'name': 'image-1'}
        extra_props = {'x_none_read': 'bar'}
        created_image = self.controller.create(request, image=image, extra_properties=extra_props, tags=[])
        another_request = unit_test_utils.get_fake_request(roles=['fake_role'])
        output = self.controller.show(another_request, created_image.image_id)
        self.assertRaises(KeyError, output.extra_properties.__getitem__, 'x_none_read')

    def test_update_locked_down_protected_prop(self):
        """Property marked with special char ! updatable by no one"""
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['admin'])
        image = {'name': 'image-1'}
        extra_props = {'x_none_update': 'bar'}
        created_image = self.controller.create(request, image=image, extra_properties=extra_props, tags=[])
        another_request = unit_test_utils.get_fake_request(roles=['member', 'fake_role'])
        changes = [{'op': 'replace', 'path': ['x_none_update'], 'value': 'baz'}]
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, another_request, created_image.image_id, changes)

    def test_delete_locked_down_protected_prop(self):
        """Property marked with special char ! deletable by no one"""
        self.set_property_protections()
        request = unit_test_utils.get_fake_request(roles=['admin'])
        image = {'name': 'image-1'}
        extra_props = {'x_none_delete': 'bar'}
        created_image = self.controller.create(request, image=image, extra_properties=extra_props, tags=[])
        another_request = unit_test_utils.get_fake_request(roles=['member', 'fake_role'])
        changes = [{'op': 'remove', 'path': ['x_none_delete']}]
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, another_request, created_image.image_id, changes)

    def test_update_replace_locations_non_empty(self):
        self.config(show_multiple_locations=True)
        new_location = {'url': '%s/fake_location' % BASE_URI, 'metadata': {}}
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'replace', 'path': ['locations'], 'value': [new_location]}]
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.update, request, UUID1, changes)

    def test_update_replace_locations_metadata_update(self):
        self.config(show_multiple_locations=True)
        location = {'url': '%s/%s' % (BASE_URI, UUID1), 'metadata': {'a': 1}}
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'replace', 'path': ['locations'], 'value': [location]}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual({'a': 1}, output.locations[0]['metadata'])

    def test_locations_actions_with_locations_invisible(self):
        self.config(show_multiple_locations=False)
        new_location = {'url': '%s/fake_location' % BASE_URI, 'metadata': {}}
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'replace', 'path': ['locations'], 'value': [new_location]}]
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, UUID1, changes)

    def test_update_replace_locations_invalid(self):
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'replace', 'path': ['locations'], 'value': []}]
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, UUID1, changes)

    def test_update_add_property(self):
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['foo'], 'value': 'baz'}, {'op': 'add', 'path': ['snitch'], 'value': 'golden'}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(UUID1, output.image_id)
        self.assertEqual('baz', output.extra_properties['foo'])
        self.assertEqual('golden', output.extra_properties['snitch'])
        self.assertNotEqual(output.created_at, output.updated_at)

    def test_update_add_base_property_json_schema_version_4(self):
        request = unit_test_utils.get_fake_request()
        changes = [{'json_schema_version': 4, 'op': 'add', 'path': ['name'], 'value': 'fedora'}]
        self.assertRaises(webob.exc.HTTPConflict, self.controller.update, request, UUID1, changes)

    def test_update_add_extra_property_json_schema_version_4(self):
        self.db.image_update(None, UUID1, {'properties': {'foo': 'bar'}})
        request = unit_test_utils.get_fake_request()
        changes = [{'json_schema_version': 4, 'op': 'add', 'path': ['foo'], 'value': 'baz'}]
        self.assertRaises(webob.exc.HTTPConflict, self.controller.update, request, UUID1, changes)

    def test_update_add_base_property_json_schema_version_10(self):
        request = unit_test_utils.get_fake_request()
        changes = [{'json_schema_version': 10, 'op': 'add', 'path': ['name'], 'value': 'fedora'}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(UUID1, output.image_id)
        self.assertEqual('fedora', output.name)

    def test_update_add_extra_property_json_schema_version_10(self):
        self.db.image_update(None, UUID1, {'properties': {'foo': 'bar'}})
        request = unit_test_utils.get_fake_request()
        changes = [{'json_schema_version': 10, 'op': 'add', 'path': ['foo'], 'value': 'baz'}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(UUID1, output.image_id)
        self.assertEqual({'foo': 'baz'}, output.extra_properties)

    def test_update_add_property_already_present_json_schema_version_4(self):
        request = unit_test_utils.get_fake_request()
        properties = {'foo': 'bar'}
        self.db.image_update(None, UUID1, {'properties': properties})
        output = self.controller.show(request, UUID1)
        self.assertEqual('bar', output.extra_properties['foo'])
        changes = [{'json_schema_version': 4, 'op': 'add', 'path': ['foo'], 'value': 'baz'}]
        self.assertRaises(webob.exc.HTTPConflict, self.controller.update, request, UUID1, changes)

    def test_update_add_property_already_present_json_schema_version_10(self):
        request = unit_test_utils.get_fake_request()
        properties = {'foo': 'bar'}
        self.db.image_update(None, UUID1, {'properties': properties})
        output = self.controller.show(request, UUID1)
        self.assertEqual('bar', output.extra_properties['foo'])
        changes = [{'json_schema_version': 10, 'op': 'add', 'path': ['foo'], 'value': 'baz'}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(UUID1, output.image_id)
        self.assertEqual({'foo': 'baz'}, output.extra_properties)

    def test_update_add_locations(self):
        self.config(show_multiple_locations=True)
        new_location = {'url': '%s/fake_location' % BASE_URI, 'metadata': {}}
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': new_location}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(UUID1, output.image_id)
        self.assertEqual(2, len(output.locations))
        self.assertEqual(new_location, output.locations[1])

    @mock.patch.object(glance.quota, '_calc_required_size')
    @mock.patch.object(glance.location, '_check_image_location')
    @mock.patch.object(glance.location.ImageRepoProxy, '_set_acls')
    @mock.patch.object(store, 'get_size_from_uri_and_backend')
    @mock.patch.object(store, 'get_size_from_backend')
    def test_replace_locations_on_queued(self, mock_get_size, mock_get_size_uri, mock_set_acls, mock_check_loc, mock_calc):
        mock_calc.return_value = 1
        mock_get_size.return_value = 1
        mock_get_size_uri.return_value = 1
        self.config(show_multiple_locations=True)
        image_id = str(uuid.uuid4())
        self.images = [_db_fixture(image_id, owner=TENANT1, name='1', disk_format='raw', container_format='bare', status='queued', checksum=None, os_hash_algo=None, os_hash_value=None)]
        self.db.image_create(None, self.images[0])
        request = unit_test_utils.get_fake_request()
        new_location1 = {'url': '%s/fake_location_1' % BASE_URI, 'metadata': {}, 'validation_data': {'checksum': CHKSUM, 'os_hash_algo': 'sha512', 'os_hash_value': MULTIHASH1}}
        new_location2 = {'url': '%s/fake_location_2' % BASE_URI, 'metadata': {}, 'validation_data': {'checksum': CHKSUM, 'os_hash_algo': 'sha512', 'os_hash_value': MULTIHASH1}}
        changes = [{'op': 'replace', 'path': ['locations'], 'value': [new_location1, new_location2]}]
        output = self.controller.update(request, image_id, changes)
        self.assertEqual(image_id, output.image_id)
        self.assertEqual(2, len(output.locations))
        self.assertEqual(new_location1['url'], output.locations[0]['url'])
        self.assertEqual(new_location2['url'], output.locations[1]['url'])
        self.assertEqual('active', output.status)
        self.assertEqual(CHKSUM, output.checksum)
        self.assertEqual('sha512', output.os_hash_algo)
        self.assertEqual(MULTIHASH1, output.os_hash_value)

    @mock.patch.object(glance.quota, '_calc_required_size')
    @mock.patch.object(glance.location, '_check_image_location')
    @mock.patch.object(glance.location.ImageRepoProxy, '_set_acls')
    @mock.patch.object(store, 'get_size_from_uri_and_backend')
    @mock.patch.object(store, 'get_size_from_backend')
    def test_replace_locations_identify_associated_store(self, mock_get_size, mock_get_size_uri, mock_set_acls, mock_check_loc, mock_calc):
        mock_calc.return_value = 1
        mock_get_size.return_value = 1
        mock_get_size_uri.return_value = 1
        self.config(show_multiple_locations=True)
        self.config(enabled_backends={'fake-store': 'http'})
        image_id = str(uuid.uuid4())
        self.images = [_db_fixture(image_id, owner=TENANT1, name='1', disk_format='raw', container_format='bare', status='queued', checksum=None, os_hash_algo=None, os_hash_value=None)]
        self.db.image_create(None, self.images[0])
        request = unit_test_utils.get_fake_request()
        new_location1 = {'url': '%s/fake_location_1' % BASE_URI, 'metadata': {}, 'validation_data': {'checksum': CHKSUM, 'os_hash_algo': 'sha512', 'os_hash_value': MULTIHASH1}}
        new_location2 = {'url': '%s/fake_location_2' % BASE_URI, 'metadata': {}, 'validation_data': {'checksum': CHKSUM, 'os_hash_algo': 'sha512', 'os_hash_value': MULTIHASH1}}
        changes = [{'op': 'replace', 'path': ['locations'], 'value': [new_location1, new_location2]}]
        with mock.patch.object(store_utils, '_get_store_id_from_uri') as mock_store:
            mock_store.return_value = 'fake-store'
            new_location1['metadata']['store'] = 'fake-store'
            new_location1['metadata']['store'] = 'fake-store'
            output = self.controller.update(request, image_id, changes)
            self.assertEqual(2, len(output.locations))
            self.assertEqual(image_id, output.image_id)
            self.assertEqual(new_location1, output.locations[0])
            self.assertEqual(new_location2, output.locations[1])
            self.assertEqual('active', output.status)
            self.assertEqual(CHKSUM, output.checksum)
            self.assertEqual('sha512', output.os_hash_algo)
            self.assertEqual(MULTIHASH1, output.os_hash_value)

    @mock.patch.object(glance.quota, '_calc_required_size')
    @mock.patch.object(glance.location, '_check_image_location')
    @mock.patch.object(glance.location.ImageRepoProxy, '_set_acls')
    @mock.patch.object(store, 'get_size_from_uri_and_backend')
    @mock.patch.object(store, 'get_size_from_backend')
    def test_replace_locations_unknon_locations(self, mock_get_size, mock_get_size_uri, mock_set_acls, mock_check_loc, mock_calc):
        mock_calc.return_value = 1
        mock_get_size.return_value = 1
        mock_get_size_uri.return_value = 1
        self.config(show_multiple_locations=True)
        self.config(enabled_backends={'fake-store': 'http'})
        image_id = str(uuid.uuid4())
        self.images = [_db_fixture(image_id, owner=TENANT1, name='1', disk_format='raw', container_format='bare', status='queued', checksum=None, os_hash_algo=None, os_hash_value=None)]
        self.db.image_create(None, self.images[0])
        request = unit_test_utils.get_fake_request()
        new_location1 = {'url': 'unknown://whocares', 'metadata': {}, 'validation_data': {'checksum': CHKSUM, 'os_hash_algo': 'sha512', 'os_hash_value': MULTIHASH1}}
        new_location2 = {'url': 'unknown://whatever', 'metadata': {'store': 'unkstore'}, 'validation_data': {'checksum': CHKSUM, 'os_hash_algo': 'sha512', 'os_hash_value': MULTIHASH1}}
        changes = [{'op': 'replace', 'path': ['locations'], 'value': [new_location1, new_location2]}]
        output = self.controller.update(request, image_id, changes)
        self.assertEqual(2, len(output.locations))
        self.assertEqual(image_id, output.image_id)
        self.assertEqual('active', output.status)
        self.assertEqual(CHKSUM, output.checksum)
        self.assertEqual('sha512', output.os_hash_algo)
        self.assertEqual(MULTIHASH1, output.os_hash_value)
        self.assertEqual(new_location1, output.locations[0])
        self.assertEqual(new_location2, output.locations[1])

    @mock.patch.object(glance.quota, '_calc_required_size')
    @mock.patch.object(glance.location, '_check_image_location')
    @mock.patch.object(glance.location.ImageRepoProxy, '_set_acls')
    @mock.patch.object(store, 'get_size_from_uri_and_backend')
    @mock.patch.object(store, 'get_size_from_backend')
    def test_add_location_new_validation_data_on_active(self, mock_get_size, mock_get_size_uri, mock_set_acls, mock_check_loc, mock_calc):
        mock_calc.return_value = 1
        mock_get_size.return_value = 1
        mock_get_size_uri.return_value = 1
        self.config(show_multiple_locations=True)
        image_id = str(uuid.uuid4())
        self.images = [_db_fixture(image_id, owner=TENANT1, name='1', disk_format='raw', container_format='bare', status='active', checksum=None, os_hash_algo=None, os_hash_value=None)]
        self.db.image_create(None, self.images[0])
        request = unit_test_utils.get_fake_request()
        new_location = {'url': '%s/fake_location_1' % BASE_URI, 'metadata': {}, 'validation_data': {'checksum': CHKSUM, 'os_hash_algo': 'sha512', 'os_hash_value': MULTIHASH1}}
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': new_location}]
        self.assertRaisesRegex(webob.exc.HTTPConflict, "may only be provided when image status is 'queued'", self.controller.update, request, image_id, changes)

    @mock.patch.object(glance.quota, '_calc_required_size')
    @mock.patch.object(glance.location, '_check_image_location')
    @mock.patch.object(glance.location.ImageRepoProxy, '_set_acls')
    @mock.patch.object(store, 'get_size_from_uri_and_backend')
    @mock.patch.object(store, 'get_size_from_backend')
    def test_replace_locations_different_validation_data(self, mock_get_size, mock_get_size_uri, mock_set_acls, mock_check_loc, mock_calc):
        mock_calc.return_value = 1
        mock_get_size.return_value = 1
        mock_get_size_uri.return_value = 1
        self.config(show_multiple_locations=True)
        image_id = str(uuid.uuid4())
        self.images = [_db_fixture(image_id, owner=TENANT1, name='1', disk_format='raw', container_format='bare', status='active', checksum=CHKSUM, os_hash_algo='sha512', os_hash_value=MULTIHASH1)]
        self.db.image_create(None, self.images[0])
        request = unit_test_utils.get_fake_request()
        new_location = {'url': '%s/fake_location_1' % BASE_URI, 'metadata': {}, 'validation_data': {'checksum': CHKSUM1, 'os_hash_algo': 'sha512', 'os_hash_value': MULTIHASH2}}
        changes = [{'op': 'replace', 'path': ['locations'], 'value': [new_location]}]
        self.assertRaisesRegex(webob.exc.HTTPConflict, 'already set with a different value', self.controller.update, request, image_id, changes)

    @mock.patch.object(glance.quota, '_calc_required_size')
    @mock.patch.object(glance.location, '_check_image_location')
    @mock.patch.object(glance.location.ImageRepoProxy, '_set_acls')
    @mock.patch.object(store, 'get_size_from_uri_and_backend')
    @mock.patch.object(store, 'get_size_from_backend')
    def _test_add_location_on_queued(self, visibility, mock_get_size, mock_get_size_uri, mock_set_acls, mock_check_loc, mock_calc):
        mock_calc.return_value = 1
        mock_get_size.return_value = 1
        mock_get_size_uri.return_value = 1
        self.config(show_multiple_locations=True)
        image_id = str(uuid.uuid4())
        self.images = [_db_fixture(image_id, owner=TENANT1, checksum=CHKSUM, name='1', disk_format='raw', container_format='bare', visibility=visibility, status='queued')]
        self.db.image_create(None, self.images[0])
        request = unit_test_utils.get_fake_request()
        new_location = {'url': '%s/fake_location_1' % BASE_URI, 'metadata': {}}
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': new_location}]
        output = self.controller.update(request, image_id, changes)
        self.assertEqual(image_id, output.image_id)
        self.assertEqual(1, len(output.locations))
        self.assertEqual(new_location, output.locations[0])
        self.assertEqual('active', output.status)
        self.assertEqual(visibility, output.visibility)
        mock_set_acls.assert_called_once()

    def test_add_location_on_queued_shared(self):
        self._test_add_location_on_queued('shared')

    def test_add_location_on_queued_community(self):
        self._test_add_location_on_queued('community')

    def test_add_location_on_queued_public(self):
        self._test_add_location_on_queued('public')

    @mock.patch.object(glance.quota, '_calc_required_size')
    @mock.patch.object(glance.location, '_check_image_location')
    @mock.patch.object(glance.location.ImageRepoProxy, '_set_acls')
    @mock.patch.object(store, 'get_size_from_uri_and_backend')
    @mock.patch.object(store, 'get_size_from_backend')
    def test_add_location_identify_associated_store(self, mock_get_size, mock_get_size_uri, mock_set_acls, mock_check_loc, mock_calc):
        mock_calc.return_value = 1
        mock_get_size.return_value = 1
        mock_get_size_uri.return_value = 1
        self.config(show_multiple_locations=True)
        self.config(enabled_backends={'fake-store': 'http'})
        image_id = str(uuid.uuid4())
        self.images = [_db_fixture(image_id, owner=TENANT1, checksum=CHKSUM, name='1', disk_format='raw', container_format='bare', status='queued')]
        self.db.image_create(None, self.images[0])
        request = unit_test_utils.get_fake_request()
        new_location = {'url': '%s/fake_location_1' % BASE_URI, 'metadata': {}}
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': new_location}]
        with mock.patch.object(store_utils, '_get_store_id_from_uri') as mock_store:
            mock_store.return_value = 'fake-store'
            output = self.controller.update(request, image_id, changes)
            self.assertEqual(image_id, output.image_id)
            self.assertEqual(1, len(output.locations))
            self.assertEqual('active', output.status)
            new_location['metadata']['store'] = 'fake-store'
            self.assertEqual(new_location, output.locations[0])

    @mock.patch.object(glance.quota, '_calc_required_size')
    @mock.patch.object(glance.location, '_check_image_location')
    @mock.patch.object(glance.location.ImageRepoProxy, '_set_acls')
    @mock.patch.object(store, 'get_size_from_uri_and_backend')
    @mock.patch.object(store, 'get_size_from_backend')
    def test_add_location_unknown_locations(self, mock_get_size, mock_get_size_uri, mock_set_acls, mock_check_loc, mock_calc):
        mock_calc.return_value = 1
        mock_get_size.return_value = 1
        mock_get_size_uri.return_value = 1
        self.config(show_multiple_locations=True)
        self.config(enabled_backends={'fake-store': 'http'})
        image_id = str(uuid.uuid4())
        self.images = [_db_fixture(image_id, owner=TENANT1, checksum=CHKSUM, name='1', disk_format='raw', container_format='bare', status='queued')]
        self.db.image_create(None, self.images[0])
        new_location = {'url': 'unknown://whocares', 'metadata': {}}
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': new_location}]
        output = self.controller.update(request, image_id, changes)
        self.assertEqual(image_id, output.image_id)
        self.assertEqual('active', output.status)
        self.assertEqual(1, len(output.locations))
        self.assertEqual(new_location, output.locations[0])

    @mock.patch.object(glance.quota, '_calc_required_size')
    @mock.patch.object(glance.location, '_check_image_location')
    @mock.patch.object(glance.location.ImageRepoProxy, '_set_acls')
    @mock.patch.object(store, 'get_size_from_uri_and_backend')
    @mock.patch.object(store, 'get_size_from_backend')
    def test_add_location_invalid_validation_data(self, mock_get_size, mock_get_size_uri, mock_set_acls, mock_check_loc, mock_calc):
        mock_calc.return_value = 1
        mock_get_size.return_value = 1
        mock_get_size_uri.return_value = 1
        self.config(show_multiple_locations=True)
        image_id = str(uuid.uuid4())
        self.images = [_db_fixture(image_id, owner=TENANT1, checksum=None, os_hash_algo=None, os_hash_value=None, name='1', disk_format='raw', container_format='bare', status='queued')]
        self.db.image_create(None, self.images[0])
        request = unit_test_utils.get_fake_request()
        location = {'url': '%s/fake_location_1' % BASE_URI, 'metadata': {}, 'validation_data': {}}
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': location}]
        changes[0]['value']['validation_data'] = {'checksum': 'something the same length as md5', 'os_hash_algo': 'sha512', 'os_hash_value': MULTIHASH1}
        self.assertRaisesRegex(webob.exc.HTTPConflict, 'checksum .* is not a valid hexadecimal value', self.controller.update, request, image_id, changes)
        changes[0]['value']['validation_data'] = {'checksum': '0123456789abcdef', 'os_hash_algo': 'sha512', 'os_hash_value': MULTIHASH1}
        self.assertRaisesRegex(webob.exc.HTTPConflict, 'checksum .* is not the correct size', self.controller.update, request, image_id, changes)
        changes[0]['value']['validation_data'] = {'checksum': CHKSUM, 'os_hash_algo': 'sha256', 'os_hash_value': MULTIHASH1}
        self.assertRaisesRegex(webob.exc.HTTPConflict, 'os_hash_algo must be sha512', self.controller.update, request, image_id, changes)
        changes[0]['value']['validation_data'] = {'checksum': CHKSUM, 'os_hash_algo': 'sha512', 'os_hash_value': 'not a hex value'}
        self.assertRaisesRegex(webob.exc.HTTPConflict, 'os_hash_value .* is not a valid hexadecimal value', self.controller.update, request, image_id, changes)
        changes[0]['value']['validation_data'] = {'checksum': CHKSUM, 'os_hash_algo': 'sha512', 'os_hash_value': '0123456789abcdef'}
        self.assertRaisesRegex(webob.exc.HTTPConflict, 'os_hash_value .* is not the correct size for sha512', self.controller.update, request, image_id, changes)

    @mock.patch.object(glance.quota, '_calc_required_size')
    @mock.patch.object(glance.location, '_check_image_location')
    @mock.patch.object(glance.location.ImageRepoProxy, '_set_acls')
    @mock.patch.object(store, 'get_size_from_uri_and_backend')
    @mock.patch.object(store, 'get_size_from_backend')
    def test_add_location_same_validation_data(self, mock_get_size, mock_get_size_uri, mock_set_acls, mock_check_loc, mock_calc):
        mock_calc.return_value = 1
        mock_get_size.return_value = 1
        mock_get_size_uri.return_value = 1
        self.config(show_multiple_locations=True)
        image_id = str(uuid.uuid4())
        os_hash_value = '6513f21e44aa3da349f248188a44bc304a3653a04122d8fb4535423c8e1d14cd6a153f735bb0982e2161b5b5186106570c17a9e58b64dd39390617cd5a350f78'
        self.images = [_db_fixture(image_id, owner=TENANT1, name='1', disk_format='raw', container_format='bare', status='active', checksum='checksum1', os_hash_algo='sha512', os_hash_value=os_hash_value)]
        self.db.image_create(None, self.images[0])
        request = unit_test_utils.get_fake_request()
        new_location = {'url': '%s/fake_location_1' % BASE_URI, 'metadata': {}, 'validation_data': {'checksum': 'checksum1', 'os_hash_algo': 'sha512', 'os_hash_value': os_hash_value}}
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': new_location}]
        output = self.controller.update(request, image_id, changes)
        self.assertEqual(image_id, output.image_id)
        self.assertEqual(1, len(output.locations))
        self.assertEqual(new_location, output.locations[0])
        self.assertEqual('active', output.status)

    @mock.patch.object(glance.quota, '_calc_required_size')
    @mock.patch.object(glance.location, '_check_image_location')
    @mock.patch.object(glance.location.ImageRepoProxy, '_set_acls')
    @mock.patch.object(store, 'get_size_from_uri_and_backend')
    @mock.patch.object(store, 'get_size_from_backend')
    def test_add_location_different_validation_data(self, mock_get_size, mock_get_size_uri, mock_set_acls, mock_check_loc, mock_calc):
        mock_calc.return_value = 1
        mock_get_size.return_value = 1
        mock_get_size_uri.return_value = 1
        self.config(show_multiple_locations=True)
        image_id = str(uuid.uuid4())
        self.images = [_db_fixture(image_id, owner=TENANT1, name='1', disk_format='raw', container_format='bare', status='active', checksum=CHKSUM, os_hash_algo='sha512', os_hash_value=MULTIHASH1)]
        self.db.image_create(None, self.images[0])
        request = unit_test_utils.get_fake_request()
        new_location = {'url': '%s/fake_location_1' % BASE_URI, 'metadata': {}, 'validation_data': {'checksum': CHKSUM1, 'os_hash_algo': 'sha512', 'os_hash_value': MULTIHASH2}}
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': new_location}]
        self.assertRaisesRegex(webob.exc.HTTPConflict, 'already set with a different value', self.controller.update, request, image_id, changes)

    def _test_update_locations_status(self, image_status, update):
        self.config(show_multiple_locations=True)
        self.images = [_db_fixture('1', owner=TENANT1, checksum=CHKSUM, name='1', disk_format='raw', container_format='bare', status=image_status)]
        request = unit_test_utils.get_fake_request()
        if image_status == 'deactivated':
            self.db.image_create(request.context, self.images[0])
        else:
            self.db.image_create(None, self.images[0])
        new_location = {'url': '%s/fake_location' % BASE_URI, 'metadata': {}}
        changes = [{'op': update, 'path': ['locations', '-'], 'value': new_location}]
        self.assertRaises(webob.exc.HTTPConflict, self.controller.update, request, '1', changes)

    def test_location_add_not_permitted_status_saving(self):
        self._test_update_locations_status('saving', 'add')

    def test_location_add_not_permitted_status_deactivated(self):
        self._test_update_locations_status('deactivated', 'add')

    def test_location_add_not_permitted_status_deleted(self):
        self._test_update_locations_status('deleted', 'add')

    def test_location_add_not_permitted_status_pending_delete(self):
        self._test_update_locations_status('pending_delete', 'add')

    def test_location_add_not_permitted_status_killed(self):
        self._test_update_locations_status('killed', 'add')

    def test_location_add_not_permitted_status_importing(self):
        self._test_update_locations_status('importing', 'add')

    def test_location_add_not_permitted_status_uploading(self):
        self._test_update_locations_status('uploading', 'add')

    def test_location_remove_not_permitted_status_saving(self):
        self._test_update_locations_status('saving', 'remove')

    def test_location_remove_not_permitted_status_deactivated(self):
        self._test_update_locations_status('deactivated', 'remove')

    def test_location_remove_not_permitted_status_deleted(self):
        self._test_update_locations_status('deleted', 'remove')

    def test_location_remove_not_permitted_status_pending_delete(self):
        self._test_update_locations_status('pending_delete', 'remove')

    def test_location_remove_not_permitted_status_killed(self):
        self._test_update_locations_status('killed', 'remove')

    def test_location_remove_not_permitted_status_queued(self):
        self._test_update_locations_status('queued', 'remove')

    def test_location_remove_not_permitted_status_importing(self):
        self._test_update_locations_status('importing', 'remove')

    def test_location_remove_not_permitted_status_uploading(self):
        self._test_update_locations_status('uploading', 'remove')

    def test_location_replace_not_permitted_status_saving(self):
        self._test_update_locations_status('saving', 'replace')

    def test_location_replace_not_permitted_status_deactivated(self):
        self._test_update_locations_status('deactivated', 'replace')

    def test_location_replace_not_permitted_status_deleted(self):
        self._test_update_locations_status('deleted', 'replace')

    def test_location_replace_not_permitted_status_pending_delete(self):
        self._test_update_locations_status('pending_delete', 'replace')

    def test_location_replace_not_permitted_status_killed(self):
        self._test_update_locations_status('killed', 'replace')

    def test_location_replace_not_permitted_status_importing(self):
        self._test_update_locations_status('importing', 'replace')

    def test_location_replace_not_permitted_status_uploading(self):
        self._test_update_locations_status('uploading', 'replace')

    def test_update_add_locations_insertion(self):
        self.config(show_multiple_locations=True)
        new_location = {'url': '%s/fake_location' % BASE_URI, 'metadata': {}}
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['locations', '0'], 'value': new_location}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(UUID1, output.image_id)
        self.assertEqual(2, len(output.locations))
        self.assertEqual(new_location, output.locations[0])

    def test_update_add_locations_list(self):
        self.config(show_multiple_locations=True)
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': {'url': 'foo', 'metadata': {}}}]
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.update, request, UUID1, changes)

    def test_update_add_locations_invalid(self):
        self.config(show_multiple_locations=True)
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': {'url': 'unknow://foo', 'metadata': {}}}]
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.update, request, UUID1, changes)
        changes = [{'op': 'add', 'path': ['locations', None], 'value': {'url': 'unknow://foo', 'metadata': {}}}]
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.update, request, UUID1, changes)

    def test_update_add_duplicate_locations(self):
        self.config(show_multiple_locations=True)
        new_location = {'url': '%s/fake_location' % BASE_URI, 'metadata': {}}
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': new_location}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(UUID1, output.image_id)
        self.assertEqual(2, len(output.locations))
        self.assertEqual(new_location, output.locations[1])
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.update, request, UUID1, changes)

    def test_update_add_too_many_locations(self):
        self.config(show_multiple_locations=True)
        self.config(image_location_quota=1)
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': {'url': '%s/fake_location_1' % BASE_URI, 'metadata': {}}}, {'op': 'add', 'path': ['locations', '-'], 'value': {'url': '%s/fake_location_2' % BASE_URI, 'metadata': {}}}]
        self.assertRaises(webob.exc.HTTPRequestEntityTooLarge, self.controller.update, request, UUID1, changes)

    def test_update_add_and_remove_too_many_locations(self):
        self.config(show_multiple_locations=True)
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': {'url': '%s/fake_location_1' % BASE_URI, 'metadata': {}}}, {'op': 'add', 'path': ['locations', '-'], 'value': {'url': '%s/fake_location_2' % BASE_URI, 'metadata': {}}}]
        self.controller.update(request, UUID1, changes)
        self.config(image_location_quota=1)
        changes = [{'op': 'remove', 'path': ['locations', '0']}, {'op': 'add', 'path': ['locations', '-'], 'value': {'url': '%s/fake_location_3' % BASE_URI, 'metadata': {}}}]
        self.assertRaises(webob.exc.HTTPRequestEntityTooLarge, self.controller.update, request, UUID1, changes)

    def test_update_add_unlimited_locations(self):
        self.config(show_multiple_locations=True)
        self.config(image_location_quota=-1)
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': {'url': '%s/fake_location_1' % BASE_URI, 'metadata': {}}}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(UUID1, output.image_id)
        self.assertNotEqual(output.created_at, output.updated_at)

    def test_update_remove_location_while_over_limit(self):
        """Ensure that image locations can be removed.

        Image locations should be able to be removed as long as the image has
        fewer than the limited number of image locations after the
        transaction.
        """
        self.config(show_multiple_locations=True)
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': {'url': '%s/fake_location_1' % BASE_URI, 'metadata': {}}}, {'op': 'add', 'path': ['locations', '-'], 'value': {'url': '%s/fake_location_2' % BASE_URI, 'metadata': {}}}]
        self.controller.update(request, UUID1, changes)
        self.config(image_location_quota=1)
        self.config(show_multiple_locations=True)
        changes = [{'op': 'remove', 'path': ['locations', '0']}, {'op': 'remove', 'path': ['locations', '0']}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(UUID1, output.image_id)
        self.assertEqual(1, len(output.locations))
        self.assertIn('fake_location_2', output.locations[0]['url'])
        self.assertNotEqual(output.created_at, output.updated_at)

    def test_update_add_and_remove_location_under_limit(self):
        """Ensure that image locations can be removed.

        Image locations should be able to be added and removed simultaneously
        as long as the image has fewer than the limited number of image
        locations after the transaction.
        """
        self.mock_object(store, 'get_size_from_backend', unit_test_utils.fake_get_size_from_backend)
        self.config(show_multiple_locations=True)
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': {'url': '%s/fake_location_1' % BASE_URI, 'metadata': {}}}, {'op': 'add', 'path': ['locations', '-'], 'value': {'url': '%s/fake_location_2' % BASE_URI, 'metadata': {}}}]
        self.controller.update(request, UUID1, changes)
        self.config(image_location_quota=2)
        changes = [{'op': 'remove', 'path': ['locations', '0']}, {'op': 'remove', 'path': ['locations', '0']}, {'op': 'add', 'path': ['locations', '-'], 'value': {'url': '%s/fake_location_3' % BASE_URI, 'metadata': {}}}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(UUID1, output.image_id)
        self.assertEqual(2, len(output.locations))
        self.assertIn('fake_location_3', output.locations[1]['url'])
        self.assertNotEqual(output.created_at, output.updated_at)

    def test_update_remove_base_property(self):
        self.db.image_update(None, UUID1, {'properties': {'foo': 'bar'}})
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'remove', 'path': ['name']}]
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, UUID1, changes)

    def test_update_remove_property(self):
        request = unit_test_utils.get_fake_request()
        properties = {'foo': 'bar', 'snitch': 'golden'}
        self.db.image_update(None, UUID1, {'properties': properties})
        output = self.controller.show(request, UUID1)
        self.assertEqual('bar', output.extra_properties['foo'])
        self.assertEqual('golden', output.extra_properties['snitch'])
        changes = [{'op': 'remove', 'path': ['snitch']}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(UUID1, output.image_id)
        self.assertEqual({'foo': 'bar'}, output.extra_properties)
        self.assertNotEqual(output.created_at, output.updated_at)

    def test_update_remove_missing_property(self):
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'remove', 'path': ['foo']}]
        self.assertRaises(webob.exc.HTTPConflict, self.controller.update, request, UUID1, changes)

    def test_update_remove_location(self):
        self.config(show_multiple_locations=True)
        self.mock_object(store, 'get_size_from_backend', unit_test_utils.fake_get_size_from_backend)
        request = unit_test_utils.get_fake_request()
        new_location = {'url': '%s/fake_location' % BASE_URI, 'metadata': {}}
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': new_location}]
        self.controller.update(request, UUID1, changes)
        changes = [{'op': 'remove', 'path': ['locations', '0']}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(UUID1, output.image_id)
        self.assertEqual(1, len(output.locations))
        self.assertEqual('active', output.status)

    def test_update_remove_location_invalid_pos(self):
        self.config(show_multiple_locations=True)
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': {'url': '%s/fake_location' % BASE_URI, 'metadata': {}}}]
        self.controller.update(request, UUID1, changes)
        changes = [{'op': 'remove', 'path': ['locations', None]}]
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.update, request, UUID1, changes)
        changes = [{'op': 'remove', 'path': ['locations', '-1']}]
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.update, request, UUID1, changes)
        changes = [{'op': 'remove', 'path': ['locations', '99']}]
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.update, request, UUID1, changes)
        changes = [{'op': 'remove', 'path': ['locations', 'x']}]
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.update, request, UUID1, changes)

    def test_update_remove_location_store_exception(self):
        self.config(show_multiple_locations=True)

        def fake_delete_image_location_from_backend(self, *args, **kwargs):
            raise Exception('fake_backend_exception')
        self.mock_object(self.store_utils, 'delete_image_location_from_backend', fake_delete_image_location_from_backend)
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': {'url': '%s/fake_location' % BASE_URI, 'metadata': {}}}]
        self.controller.update(request, UUID1, changes)
        changes = [{'op': 'remove', 'path': ['locations', '0']}]
        self.assertRaises(webob.exc.HTTPInternalServerError, self.controller.update, request, UUID1, changes)

    def test_update_multiple_changes(self):
        request = unit_test_utils.get_fake_request()
        properties = {'foo': 'bar', 'snitch': 'golden'}
        self.db.image_update(None, UUID1, {'properties': properties})
        changes = [{'op': 'replace', 'path': ['min_ram'], 'value': 128}, {'op': 'replace', 'path': ['foo'], 'value': 'baz'}, {'op': 'remove', 'path': ['snitch']}, {'op': 'add', 'path': ['kb'], 'value': 'dvorak'}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(UUID1, output.image_id)
        self.assertEqual(128, output.min_ram)
        self.addDetail('extra_properties', testtools.content.json_content(jsonutils.dumps(output.extra_properties)))
        self.assertEqual(2, len(output.extra_properties))
        self.assertEqual('baz', output.extra_properties['foo'])
        self.assertEqual('dvorak', output.extra_properties['kb'])
        self.assertNotEqual(output.created_at, output.updated_at)

    def test_update_invalid_operation(self):
        request = unit_test_utils.get_fake_request()
        change = {'op': 'test', 'path': 'options', 'value': 'puts'}
        try:
            self.controller.update(request, UUID1, [change])
        except AttributeError:
            pass
        else:
            self.fail('Failed to raise AssertionError on %s' % change)

    def test_update_duplicate_tags(self):
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'replace', 'path': ['tags'], 'value': ['ping', 'ping']}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual(1, len(output.tags))
        self.assertIn('ping', output.tags)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('image.update', output_log['event_type'])
        self.assertEqual(UUID1, output_log['payload']['id'])

    def test_update_disabled_notification(self):
        self.config(disabled_notifications=['image.update'])
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'replace', 'path': ['name'], 'value': 'Ping Pong'}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual('Ping Pong', output.name)
        output_logs = self.notifier.get_logs()
        self.assertEqual(0, len(output_logs))

    def test_delete(self):
        request = unit_test_utils.get_fake_request()
        self.assertIn('%s/%s' % (BASE_URI, UUID1), self.store.data)
        try:
            self.controller.delete(request, UUID1)
            output_logs = self.notifier.get_logs()
            self.assertEqual(1, len(output_logs))
            output_log = output_logs[0]
            self.assertEqual('INFO', output_log['notification_type'])
            self.assertEqual('image.delete', output_log['event_type'])
        except Exception as e:
            self.fail('Delete raised exception: %s' % e)
        deleted_img = self.db.image_get(request.context, UUID1, force_show_deleted=True)
        self.assertTrue(deleted_img['deleted'])
        self.assertEqual('deleted', deleted_img['status'])
        self.assertNotIn('%s/%s' % (BASE_URI, UUID1), self.store.data)

    def test_delete_not_allowed_by_policy(self):
        request = unit_test_utils.get_fake_request()
        with mock.patch.object(self.controller.policy, 'enforce') as mock_enf:
            mock_enf.side_effect = webob.exc.HTTPForbidden()
            exc = self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete, request, UUID1)
            self.assertTrue(mock_enf.called)
        self.assertEqual('The resource could not be found.', str(exc))
        with mock.patch.object(self.controller.policy, 'enforce') as mock_enf:
            mock_enf.side_effect = [webob.exc.HTTPForbidden(), lambda *a: None]
            exc = self.assertRaises(webob.exc.HTTPForbidden, self.controller.delete, request, UUID1)
            self.assertTrue(mock_enf.called)

    @mock.patch.object(store, 'get_store_from_store_identifier')
    @mock.patch.object(store.location, 'get_location_from_uri_and_backend')
    @mock.patch.object(store_utils, 'get_dir_separator')
    def test_verify_staging_data_deleted_on_image_delete(self, mock_get_dir_separator, mock_location, mock_store):
        self.config(enabled_backends={'fake-store': 'file'})
        fake_staging_store = mock.Mock()
        mock_store.return_value = fake_staging_store
        mock_get_dir_separator.return_value = ('/', '/tmp/os_glance_staging_store')
        image_id = str(uuid.uuid4())
        self.images = [_db_fixture(image_id, owner=TENANT1, name='1', disk_format='raw', container_format='bare', status='importing', checksum=None, os_hash_algo=None, os_hash_value=None)]
        self.db.image_create(None, self.images[0])
        request = unit_test_utils.get_fake_request()
        try:
            self.controller.delete(request, image_id)
            self.assertEqual(1, mock_store.call_count)
            mock_store.assert_called_once_with('os_glance_staging_store')
            self.assertEqual(1, mock_location.call_count)
            fake_staging_store.delete.assert_called_once()
        except Exception as e:
            self.fail('Delete raised exception: %s' % e)
        deleted_img = self.db.image_get(request.context, image_id, force_show_deleted=True)
        self.assertTrue(deleted_img['deleted'])
        self.assertEqual('deleted', deleted_img['status'])

    def test_delete_with_tags(self):
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'replace', 'path': ['tags'], 'value': ['many', 'cool', 'new', 'tags']}]
        self.controller.update(request, UUID1, changes)
        self.assertIn('%s/%s' % (BASE_URI, UUID1), self.store.data)
        self.controller.delete(request, UUID1)
        output_logs = self.notifier.get_logs()
        output_delete_logs = [output_log for output_log in output_logs if output_log['event_type'] == 'image.delete']
        self.assertEqual(1, len(output_delete_logs))
        output_log = output_delete_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        deleted_img = self.db.image_get(request.context, UUID1, force_show_deleted=True)
        self.assertTrue(deleted_img['deleted'])
        self.assertEqual('deleted', deleted_img['status'])
        self.assertNotIn('%s/%s' % (BASE_URI, UUID1), self.store.data)

    def test_delete_disabled_notification(self):
        self.config(disabled_notifications=['image.delete'])
        request = unit_test_utils.get_fake_request()
        self.assertIn('%s/%s' % (BASE_URI, UUID1), self.store.data)
        try:
            self.controller.delete(request, UUID1)
            output_logs = self.notifier.get_logs()
            self.assertEqual(0, len(output_logs))
        except Exception as e:
            self.fail('Delete raised exception: %s' % e)
        deleted_img = self.db.image_get(request.context, UUID1, force_show_deleted=True)
        self.assertTrue(deleted_img['deleted'])
        self.assertEqual('deleted', deleted_img['status'])
        self.assertNotIn('%s/%s' % (BASE_URI, UUID1), self.store.data)

    def test_delete_queued_updates_status(self):
        """Ensure status of queued image is updated (LP bug #1048851)"""
        request = unit_test_utils.get_fake_request(is_admin=True)
        image = self.db.image_create(request.context, {'status': 'queued'})
        image_id = image['id']
        self.controller.delete(request, image_id)
        image = self.db.image_get(request.context, image_id, force_show_deleted=True)
        self.assertTrue(image['deleted'])
        self.assertEqual('deleted', image['status'])

    def test_delete_queued_updates_status_delayed_delete(self):
        """Ensure status of queued image is updated (LP bug #1048851).

        Must be set to 'deleted' when delayed_delete isenabled.
        """
        self.config(delayed_delete=True)
        request = unit_test_utils.get_fake_request(is_admin=True)
        image = self.db.image_create(request.context, {'status': 'queued'})
        image_id = image['id']
        self.controller.delete(request, image_id)
        image = self.db.image_get(request.context, image_id, force_show_deleted=True)
        self.assertTrue(image['deleted'])
        self.assertEqual('deleted', image['status'])

    def test_delete_not_in_store(self):
        request = unit_test_utils.get_fake_request()
        self.assertIn('%s/%s' % (BASE_URI, UUID1), self.store.data)
        for k in self.store.data:
            if UUID1 in k:
                del self.store.data[k]
                break
        self.controller.delete(request, UUID1)
        deleted_img = self.db.image_get(request.context, UUID1, force_show_deleted=True)
        self.assertTrue(deleted_img['deleted'])
        self.assertEqual('deleted', deleted_img['status'])
        self.assertNotIn('%s/%s' % (BASE_URI, UUID1), self.store.data)

    def test_delayed_delete(self):
        self.config(delayed_delete=True)
        request = unit_test_utils.get_fake_request()
        self.assertIn('%s/%s' % (BASE_URI, UUID1), self.store.data)
        self.controller.delete(request, UUID1)
        deleted_img = self.db.image_get(request.context, UUID1, force_show_deleted=True)
        self.assertTrue(deleted_img['deleted'])
        self.assertEqual('pending_delete', deleted_img['status'])
        self.assertIn('%s/%s' % (BASE_URI, UUID1), self.store.data)

    def test_delete_non_existent(self):
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete, request, str(uuid.uuid4()))

    def test_delete_already_deleted_image_admin(self):
        request = unit_test_utils.get_fake_request(is_admin=True)
        self.controller.delete(request, UUID1)
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete, request, UUID1)

    def test_delete_not_allowed(self):
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete, request, UUID4)

    def test_delete_in_use(self):

        def fake_safe_delete_from_backend(self, *args, **kwargs):
            raise store.exceptions.InUseByStore()
        self.mock_object(self.store_utils, 'safe_delete_from_backend', fake_safe_delete_from_backend)
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPConflict, self.controller.delete, request, UUID1)

    def test_delete_has_snapshot(self):

        def fake_safe_delete_from_backend(self, *args, **kwargs):
            raise store.exceptions.HasSnapshot()
        self.mock_object(self.store_utils, 'safe_delete_from_backend', fake_safe_delete_from_backend)
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPConflict, self.controller.delete, request, UUID1)

    def test_delete_to_unallowed_status(self):
        self.config(delayed_delete=True)
        request = unit_test_utils.get_fake_request(is_admin=True)
        self.action_controller.deactivate(request, UUID1)
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.delete, request, UUID1)

    def test_delete_uploading_status_image(self):
        """Ensure uploading image is deleted (LP bug #1733289)
        Ensure image stuck in uploading state is deleted (LP bug #1836140)
        """
        request = unit_test_utils.get_fake_request(is_admin=True)
        image = self.db.image_create(request.context, {'status': 'uploading'})
        image_id = image['id']
        with mock.patch.object(os.path, 'exists') as mock_exists:
            mock_exists.return_value = True
            with mock.patch.object(os, 'unlink') as mock_unlik:
                self.controller.delete(request, image_id)
                self.assertEqual(1, mock_exists.call_count)
                self.assertEqual(1, mock_unlik.call_count)
        image = self.db.image_get(request.context, image_id, force_show_deleted=True)
        self.assertTrue(image['deleted'])
        self.assertEqual('deleted', image['status'])

    def test_deletion_of_staging_data_failed(self):
        """Ensure uploading image is deleted (LP bug #1733289)
        Ensure image stuck in uploading state is deleted (LP bug #1836140)
        """
        request = unit_test_utils.get_fake_request(is_admin=True)
        image = self.db.image_create(request.context, {'status': 'uploading'})
        image_id = image['id']
        with mock.patch.object(os.path, 'exists') as mock_exists:
            mock_exists.return_value = False
            with mock.patch.object(os, 'unlink') as mock_unlik:
                self.controller.delete(request, image_id)
                self.assertEqual(1, mock_exists.call_count)
                self.assertEqual(0, mock_unlik.call_count)
        image = self.db.image_get(request.context, image_id, force_show_deleted=True)
        self.assertTrue(image['deleted'])
        self.assertEqual('deleted', image['status'])

    def test_delete_from_store_no_multistore(self):
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete_from_store, request, 'the IDs should', 'not matter')

    def test_index_with_invalid_marker(self):
        fake_uuid = str(uuid.uuid4())
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.index, request, marker=fake_uuid)

    def test_invalid_locations_op_pos(self):
        pos = self.controller._get_locations_op_pos(None, 2, True)
        self.assertIsNone(pos)
        pos = self.controller._get_locations_op_pos('1', None, True)
        self.assertIsNone(pos)

    @mock.patch('glance.db.simple.api.image_set_property_atomic')
    @mock.patch.object(glance.notifier.TaskFactoryProxy, 'new_task')
    @mock.patch.object(glance.domain.TaskExecutorFactory, 'new_task_executor')
    @mock.patch('glance.api.common.get_thread_pool')
    @mock.patch('glance.quota.keystone.enforce_image_size_total')
    def test_image_import(self, mock_enforce, mock_gtp, mock_nte, mock_nt, mock_spa):
        request = unit_test_utils.get_fake_request()
        image = FakeImage(status='uploading')
        with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = image
            output = self.controller.import_image(request, UUID4, {'method': {'name': 'glance-direct'}})
        self.assertEqual(UUID4, output)
        mock_enforce.assert_called_once_with(request.context, request.context.project_id)
        mock_spa.assert_called_once_with(UUID4, 'os_glance_import_task', mock_nt.return_value.task_id)
        mock_gtp.assert_called_once_with('tasks_pool')
        mock_gtp.return_value.spawn.assert_called_once_with(mock_nt.return_value.run, mock_nte.return_value)

    @mock.patch.object(glance.domain.TaskFactory, 'new_task')
    @mock.patch.object(glance.notifier.ImageRepoProxy, 'get')
    def test_image_import_not_allowed(self, mock_get, mock_new_task):
        enforcer = unit_test_utils.enforcer_from_rules({'get_image': '', 'modify_image': "'{0}':%(owner)s".format(TENANT2)})
        request = unit_test_utils.get_fake_request()
        self.controller.policy = enforcer
        mock_get.return_value = FakeImage(status='uploading')
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.import_image, request, UUID4, {'method': {'name': 'glance-direct'}})
        mock_new_task.assert_not_called()

    @mock.patch.object(glance.notifier.ImageRepoProxy, 'get')
    @mock.patch('glance.quota.keystone.enforce_image_size_total')
    def test_image_import_quota_fail(self, mock_enforce, mock_get):
        request = unit_test_utils.get_fake_request()
        mock_get.return_value = FakeImage(status='uploading')
        mock_enforce.side_effect = exception.LimitExceeded('test')
        self.assertRaises(webob.exc.HTTPRequestEntityTooLarge, self.controller.import_image, request, UUID4, {'method': {'name': 'glance-direct'}})

    @mock.patch('glance.db.simple.api.image_set_property_atomic')
    @mock.patch('glance.context.RequestContext.elevated')
    @mock.patch.object(glance.domain.TaskFactory, 'new_task')
    @mock.patch.object(glance.notifier.ImageRepoProxy, 'get')
    def test_image_import_copy_allowed_by_policy(self, mock_get, mock_new_task, mock_elevated, mock_spa, allowed=True):
        request = unit_test_utils.get_fake_request(tenant=TENANT2)
        mock_get.return_value = FakeImage(status='active', locations=[])
        self.policy.rules = {'copy_image': allowed}
        req_body = {'method': {'name': 'copy-image'}, 'stores': ['cheap']}
        with mock.patch.object(self.controller.gateway, 'get_task_executor_factory', side_effect=self.controller.gateway.get_task_executor_factory) as mock_tef:
            self.controller.import_image(request, UUID4, req_body)
            mock_tef.assert_called_once_with(request.context, admin_context=mock_elevated.return_value)
        expected_input = {'image_id': UUID4, 'import_req': mock.ANY, 'backend': mock.ANY}
        mock_new_task.assert_called_with(task_type='api_image_import', owner=TENANT2, task_input=expected_input, image_id=UUID4, user_id=request.context.user_id, request_id=request.context.request_id)

    def test_image_import_copy_not_allowed_by_policy(self):
        self.assertRaises(webob.exc.HTTPForbidden, self.test_image_import_copy_allowed_by_policy, allowed=False)

    @mock.patch.object(glance.notifier.ImageRepoProxy, 'get')
    def test_image_import_locked(self, mock_get):
        task = test_tasks_resource._db_fixture(test_tasks_resource.UUID1, status='pending')
        self.db.task_create(None, task)
        image = FakeImage(status='uploading')
        image.extra_properties['os_glance_import_task'] = task['id']
        mock_get.return_value = image
        request = unit_test_utils.get_fake_request(tenant=TENANT1)
        req_body = {'method': {'name': 'glance-direct'}}
        exc = self.assertRaises(webob.exc.HTTPConflict, self.controller.import_image, request, UUID1, req_body)
        self.assertEqual('Image has active task', str(exc))

    @mock.patch('glance.db.simple.api.image_set_property_atomic')
    @mock.patch('glance.db.simple.api.image_delete_property_atomic')
    @mock.patch.object(glance.notifier.TaskFactoryProxy, 'new_task')
    @mock.patch.object(glance.notifier.ImageRepoProxy, 'get')
    def test_image_import_locked_by_reaped_task(self, mock_get, mock_nt, mock_dpi, mock_spi):
        image = FakeImage(status='uploading')
        image.extra_properties['os_glance_import_task'] = 'missing'
        mock_get.return_value = image
        request = unit_test_utils.get_fake_request(tenant=TENANT1)
        req_body = {'method': {'name': 'glance-direct'}}
        mock_nt.return_value.task_id = 'mytask'
        self.controller.import_image(request, UUID1, req_body)
        mock_dpi.assert_called_once_with(image.id, 'os_glance_import_task', 'missing')
        mock_spi.assert_called_once_with(image.id, 'os_glance_import_task', 'mytask')

    @mock.patch.object(glance.notifier.ImageRepoProxy, 'save')
    @mock.patch('glance.db.simple.api.image_set_property_atomic')
    @mock.patch('glance.db.simple.api.image_delete_property_atomic')
    @mock.patch.object(glance.notifier.TaskFactoryProxy, 'new_task')
    @mock.patch.object(glance.notifier.ImageRepoProxy, 'get')
    def test_image_import_locked_by_bustable_task(self, mock_get, mock_nt, mock_dpi, mock_spi, mock_save, task_status='processing'):
        if task_status == 'processing':
            task_input = {'backend': ['store2']}
        else:
            task_input = {}
        task = test_tasks_resource._db_fixture(test_tasks_resource.UUID1, status=task_status, input=task_input)
        self.db.task_create(None, task)
        image = FakeImage(status='uploading')
        image.extra_properties['os_glance_import_task'] = task['id']
        image.extra_properties['os_glance_importing_to_stores'] = 'store2'
        mock_get.return_value = image
        request = unit_test_utils.get_fake_request(tenant=TENANT1)
        req_body = {'method': {'name': 'glance-direct'}}
        time_fixture = fixture.TimeFixture(task['updated_at'] + datetime.timedelta(minutes=10))
        self.useFixture(time_fixture)
        mock_nt.return_value.task_id = 'mytask'
        self.assertRaises(webob.exc.HTTPConflict, self.controller.import_image, request, UUID1, req_body)
        mock_dpi.assert_not_called()
        mock_spi.assert_not_called()
        mock_nt.assert_not_called()
        time_fixture.advance_time_delta(datetime.timedelta(minutes=90))
        self.controller.import_image(request, UUID1, req_body)
        mock_dpi.assert_called_once_with(image.id, 'os_glance_import_task', task['id'])
        mock_spi.assert_called_once_with(image.id, 'os_glance_import_task', 'mytask')
        if task_status == 'processing':
            self.assertNotIn('store2', image.extra_properties['os_glance_importing_to_stores'])

    def test_image_import_locked_by_bustable_terminal_task_failure(self):
        self.test_image_import_locked_by_bustable_task(task_status='failure')

    def test_image_import_locked_by_bustable_terminal_task_success(self):
        self.test_image_import_locked_by_bustable_task(task_status='success')

    def test_cleanup_stale_task_progress(self):
        img_repo = mock.MagicMock()
        image = mock.MagicMock()
        task = mock.MagicMock()
        task.task_input = {}
        image.extra_properties = {}
        self.controller._cleanup_stale_task_progress(img_repo, image, task)
        img_repo.save.assert_not_called()
        task.task_input = {'backend': []}
        self.controller._cleanup_stale_task_progress(img_repo, image, task)
        img_repo.save.assert_not_called()
        task.task_input = {'backend': ['store1', 'store2']}
        self.controller._cleanup_stale_task_progress(img_repo, image, task)
        img_repo.save.assert_not_called()
        image.extra_properties = {'os_glance_failed_import': 'store3'}
        self.controller._cleanup_stale_task_progress(img_repo, image, task)
        img_repo.save.assert_not_called()
        image.extra_properties = {'os_glance_importing_to_stores': 'foo,store1,bar', 'os_glance_failed_import': 'foo,store2,bar'}
        self.controller._cleanup_stale_task_progress(img_repo, image, task)
        img_repo.save.assert_called_once_with(image)
        self.assertEqual({'os_glance_importing_to_stores': 'foo,bar', 'os_glance_failed_import': 'foo,bar'}, image.extra_properties)

    def test_bust_import_lock_race_to_delete(self):
        image_repo = mock.MagicMock()
        task_repo = mock.MagicMock()
        image = mock.MagicMock()
        task = mock.MagicMock(id='foo')
        image_repo.delete_property_atomic.side_effect = exception.NotFound
        self.assertRaises(exception.Conflict, self.controller._bust_import_lock, image_repo, task_repo, image, task, task.id)

    def test_enforce_lock_log_not_bustable(self, task_status='processing'):
        task = test_tasks_resource._db_fixture(test_tasks_resource.UUID1, status=task_status)
        self.db.task_create(None, task)
        request = unit_test_utils.get_fake_request(tenant=TENANT1)
        image = FakeImage()
        image.extra_properties['os_glance_import_task'] = task['id']
        time_fixture = fixture.TimeFixture(task['updated_at'] + datetime.timedelta(minutes=55))
        self.useFixture(time_fixture)
        expected_expire = 300
        if task_status == 'pending':
            expected_expire += 3600
        with mock.patch.object(glance.api.v2.images, 'LOG') as mock_log:
            self.assertRaises(exception.Conflict, self.controller._enforce_import_lock, request, image)
            mock_log.warning.assert_called_once_with('Image %(image)s has active import task %(task)s in status %(status)s; lock remains valid for %(expire)i more seconds', {'image': image.id, 'task': task['id'], 'status': task_status, 'expire': expected_expire})

    def test_enforce_lock_pending_takes_longer(self):
        self.test_enforce_lock_log_not_bustable(task_status='pending')

    def test_delete_encryption_key_no_encryption_key(self):
        request = unit_test_utils.get_fake_request()
        fake_encryption_key = self.controller._key_manager.store(request.context, mock.Mock())
        image = _domain_fixture(UUID2, name='image-2', owner=TENANT2, checksum='ca425b88f047ce8ec45ee90e813ada91', os_hash_algo=FAKEHASHALGO, os_hash_value=MULTIHASH1, created_at=DATETIME, updated_at=DATETIME, size=1024, virtual_size=3072, extra_properties={})
        self.controller._delete_encryption_key(request.context, image)
        key = self.controller._key_manager.get(request.context, fake_encryption_key)
        self.assertEqual(fake_encryption_key, key._id)

    def test_delete_encryption_key_no_deletion_policy(self):
        request = unit_test_utils.get_fake_request()
        fake_encryption_key = self.controller._key_manager.store(request.context, mock.Mock())
        props = {'cinder_encryption_key_id': fake_encryption_key}
        image = _domain_fixture(UUID2, name='image-2', owner=TENANT2, checksum='ca425b88f047ce8ec45ee90e813ada91', os_hash_algo=FAKEHASHALGO, os_hash_value=MULTIHASH1, created_at=DATETIME, updated_at=DATETIME, size=1024, virtual_size=3072, extra_properties=props)
        self.controller._delete_encryption_key(request.context, image)
        key = self.controller._key_manager.get(request.context, fake_encryption_key)
        self.assertEqual(fake_encryption_key, key._id)

    def test_delete_encryption_key_do_not_delete(self):
        request = unit_test_utils.get_fake_request()
        fake_encryption_key = self.controller._key_manager.store(request.context, mock.Mock())
        props = {'cinder_encryption_key_id': fake_encryption_key, 'cinder_encryption_key_deletion_policy': 'do_not_delete'}
        image = _domain_fixture(UUID2, name='image-2', owner=TENANT2, checksum='ca425b88f047ce8ec45ee90e813ada91', os_hash_algo=FAKEHASHALGO, os_hash_value=MULTIHASH1, created_at=DATETIME, updated_at=DATETIME, size=1024, virtual_size=3072, extra_properties=props)
        self.controller._delete_encryption_key(request.context, image)
        key = self.controller._key_manager.get(request.context, fake_encryption_key)
        self.assertEqual(fake_encryption_key, key._id)

    def test_delete_encryption_key_forbidden(self):
        request = unit_test_utils.get_fake_request()
        fake_encryption_key = self.controller._key_manager.store(request.context, mock.Mock())
        props = {'cinder_encryption_key_id': fake_encryption_key, 'cinder_encryption_key_deletion_policy': 'on_image_deletion'}
        image = _domain_fixture(UUID2, name='image-2', owner=TENANT2, checksum='ca425b88f047ce8ec45ee90e813ada91', os_hash_algo=FAKEHASHALGO, os_hash_value=MULTIHASH1, created_at=DATETIME, updated_at=DATETIME, size=1024, virtual_size=3072, extra_properties=props)
        with mock.patch.object(self.controller._key_manager, 'delete', side_effect=castellan_exception.Forbidden):
            self.controller._delete_encryption_key(request.context, image)
        key = self.controller._key_manager.get(request.context, fake_encryption_key)
        self.assertEqual(fake_encryption_key, key._id)

    def test_delete_encryption_key_not_found(self):
        request = unit_test_utils.get_fake_request()
        fake_encryption_key = self.controller._key_manager.store(request.context, mock.Mock())
        props = {'cinder_encryption_key_id': fake_encryption_key, 'cinder_encryption_key_deletion_policy': 'on_image_deletion'}
        image = _domain_fixture(UUID2, name='image-2', owner=TENANT2, checksum='ca425b88f047ce8ec45ee90e813ada91', os_hash_algo=FAKEHASHALGO, os_hash_value=MULTIHASH1, created_at=DATETIME, updated_at=DATETIME, size=1024, virtual_size=3072, extra_properties=props)
        with mock.patch.object(self.controller._key_manager, 'delete', side_effect=castellan_exception.ManagedObjectNotFoundError):
            self.controller._delete_encryption_key(request.context, image)
        key = self.controller._key_manager.get(request.context, fake_encryption_key)
        self.assertEqual(fake_encryption_key, key._id)

    def test_delete_encryption_key_error(self):
        request = unit_test_utils.get_fake_request()
        fake_encryption_key = self.controller._key_manager.store(request.context, mock.Mock())
        props = {'cinder_encryption_key_id': fake_encryption_key, 'cinder_encryption_key_deletion_policy': 'on_image_deletion'}
        image = _domain_fixture(UUID2, name='image-2', owner=TENANT2, checksum='ca425b88f047ce8ec45ee90e813ada91', os_hash_algo=FAKEHASHALGO, os_hash_value=MULTIHASH1, created_at=DATETIME, updated_at=DATETIME, size=1024, virtual_size=3072, extra_properties=props)
        with mock.patch.object(self.controller._key_manager, 'delete', side_effect=castellan_exception.KeyManagerError):
            self.controller._delete_encryption_key(request.context, image)
        key = self.controller._key_manager.get(request.context, fake_encryption_key)
        self.assertEqual(fake_encryption_key, key._id)

    def test_delete_encryption_key(self):
        request = unit_test_utils.get_fake_request()
        fake_encryption_key = self.controller._key_manager.store(request.context, mock.Mock())
        props = {'cinder_encryption_key_id': fake_encryption_key, 'cinder_encryption_key_deletion_policy': 'on_image_deletion'}
        image = _domain_fixture(UUID2, name='image-2', owner=TENANT2, checksum='ca425b88f047ce8ec45ee90e813ada91', os_hash_algo=FAKEHASHALGO, os_hash_value=MULTIHASH1, created_at=DATETIME, updated_at=DATETIME, size=1024, virtual_size=3072, extra_properties=props)
        self.controller._delete_encryption_key(request.context, image)
        self.assertRaises(castellan_exception.ManagedObjectNotFoundError, self.controller._key_manager.get, request.context, fake_encryption_key)

    def test_delete_no_encryption_key_id(self):
        request = unit_test_utils.get_fake_request()
        extra_props = {'cinder_encryption_key_deletion_policy': 'on_image_deletion'}
        created_image = self.controller.create(request, image={'name': 'image-1'}, extra_properties=extra_props, tags=[])
        image_id = created_image.image_id
        self.controller.delete(request, image_id)
        image = self.db.image_get(request.context, image_id, force_show_deleted=True)
        self.assertTrue(image['deleted'])
        self.assertEqual('deleted', image['status'])

    def test_delete_invalid_encryption_key_id(self):
        request = unit_test_utils.get_fake_request()
        extra_props = {'cinder_encryption_key_id': 'invalid', 'cinder_encryption_key_deletion_policy': 'on_image_deletion'}
        created_image = self.controller.create(request, image={'name': 'image-1'}, extra_properties=extra_props, tags=[])
        image_id = created_image.image_id
        self.controller.delete(request, image_id)
        image = self.db.image_get(request.context, image_id, force_show_deleted=True)
        self.assertTrue(image['deleted'])
        self.assertEqual('deleted', image['status'])

    def test_delete_invalid_encryption_key_deletion_policy(self):
        request = unit_test_utils.get_fake_request()
        extra_props = {'cinder_encryption_key_deletion_policy': 'invalid'}
        created_image = self.controller.create(request, image={'name': 'image-1'}, extra_properties=extra_props, tags=[])
        image_id = created_image.image_id
        self.controller.delete(request, image_id)
        image = self.db.image_get(request.context, image_id, force_show_deleted=True)
        self.assertTrue(image['deleted'])
        self.assertEqual('deleted', image['status'])