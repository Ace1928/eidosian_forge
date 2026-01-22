from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
class TestImagesPolicy(functional.SynchronousAPIBase):

    def setUp(self):
        super(TestImagesPolicy, self).setUp()
        self.policy = policy.Enforcer(suppress_deprecation_warnings=True)

    def set_policy_rules(self, rules):
        self.policy.set_rules(oslo_policy.policy.Rules.from_dict(rules), overwrite=True)

    def start_server(self):
        with mock.patch.object(policy, 'Enforcer') as mock_enf:
            mock_enf.return_value = self.policy
            super(TestImagesPolicy, self).start_server()

    def test_image_update_basic(self):
        self.start_server()
        image_id = self._create_and_upload()
        resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'add', 'path': '/mykey1', 'value': 'foo'})
        self.assertEqual(200, resp.status_code, resp.text)
        self.assertEqual('foo', self.api_get('/v2/images/%s' % image_id).json['mykey1'])
        self.set_policy_rules({'get_image': '', 'modify_image': '!'})
        resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'add', 'path': '/mykey2', 'value': 'foo'})
        self.assertEqual(403, resp.status_code)
        self.assertNotIn('mykey2', self.api_get('/v2/images/%s' % image_id).json)
        resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'replace', 'path': '/mykey1', 'value': 'bar'})
        self.assertEqual(403, resp.status_code)
        self.assertEqual('foo', self.api_get('/v2/images/%s' % image_id).json['mykey1'])
        resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'remove', 'path': '/mykey1'})
        self.assertEqual(403, resp.status_code)
        self.assertEqual('foo', self.api_get('/v2/images/%s' % image_id).json['mykey1'])
        self.set_policy_rules({'get_image': '!', 'modify_image': '!'})
        resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'remove', 'path': '/mykey1'})
        self.assertEqual(404, resp.status_code)

    @mock.patch('glance.location._check_image_location', new=lambda *a: 0)
    @mock.patch('glance.location.ImageRepoProxy._set_acls', new=lambda *a: 0)
    def test_image_update_locations(self):
        self.config(show_multiple_locations=True)
        self.start_server()
        image_id = self._create_and_upload()
        resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'add', 'path': '/locations/0', 'value': {'url': 'http://foo.bar', 'metadata': {}}})
        self.assertEqual(200, resp.status_code, resp.text)
        self.assertEqual(2, len(self.api_get('/v2/images/%s' % image_id).json['locations']))
        self.assertEqual('http://foo.bar', self.api_get('/v2/images/%s' % image_id).json['locations'][1]['url'])
        resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'remove', 'path': '/locations/0'})
        self.assertEqual(200, resp.status_code, resp.text)
        self.assertEqual(1, len(self.api_get('/v2/images/%s' % image_id).json['locations']))
        resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'add', 'path': '/locations/0', 'value': {'url': 'http://foo.baz', 'metadata': {}}})
        self.assertEqual(200, resp.status_code, resp.text)
        self.assertEqual(2, len(self.api_get('/v2/images/%s' % image_id).json['locations']))
        self.set_policy_rules({'get_image': '', 'get_image_location': '', 'set_image_location': '!', 'delete_image_location': '!'})
        resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'remove', 'path': '/locations/0'})
        self.assertEqual(403, resp.status_code, resp.text)
        self.assertEqual(2, len(self.api_get('/v2/images/%s' % image_id).json['locations']))
        resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'add', 'path': '/locations/0', 'value': {'url': 'http://foo.baz', 'metadata': {}}})
        self.assertEqual(403, resp.status_code, resp.text)
        self.assertEqual(2, len(self.api_get('/v2/images/%s' % image_id).json['locations']))

    def test_image_get(self):
        self.start_server()
        image_id = self._create_and_upload()
        image = self.api_get('/v2/images/%s' % image_id).json
        self.assertEqual(image_id, image['id'])
        images = self.api_get('/v2/images').json['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        self.set_policy_rules({'get_images': '!', 'get_image': ''})
        resp = self.api_get('/v2/images')
        self.assertEqual(403, resp.status_code)
        image = self.api_get('/v2/images/%s' % image_id).json
        self.assertEqual(image_id, image['id'])
        self.set_policy_rules({'get_images': '', 'get_image': '!'})
        images = self.api_get('/v2/images').json['images']
        self.assertEqual(0, len(images))
        resp = self.api_get('/v2/images/%s' % image_id)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'get_images': '!', 'get_image': '!'})
        resp = self.api_get('/v2/images')
        self.assertEqual(403, resp.status_code)
        resp = self.api_get('/v2/images/%s' % image_id)
        self.assertEqual(404, resp.status_code)

    def test_image_create(self):
        self.start_server()
        self.assertEqual(201, self._create().status_code)
        self.set_policy_rules({'add_image': '!'})
        self.assertEqual(403, self._create().status_code)

    def test_image_create_by_another(self):
        self.start_server()
        image = {'name': 'foo', 'container_format': 'bare', 'disk_format': 'raw', 'owner': 'someoneelse'}
        resp = self.api_post('/v2/images', json=image, headers={'X-Roles': 'member'})
        self.assertIn("You are not permitted to create images owned by 'someoneelse'", resp.text)

    def test_image_delete(self):
        self.start_server()
        image_id = self._create_and_upload()
        resp = self.api_delete('/v2/images/%s' % image_id)
        self.assertEqual(204, resp.status_code)
        resp = self.api_get('/v2/images/%s' % image_id)
        self.assertEqual(404, resp.status_code)
        resp = self.api_delete('/v2/images/%s' % image_id)
        self.assertEqual(404, resp.status_code)
        image_id = self._create_and_upload()
        self.set_policy_rules({'get_image': '', 'delete_image': '!'})
        resp = self.api_delete('/v2/images/%s' % image_id)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'get_image': '!', 'delete_image': '!'})
        resp = self.api_delete('/v2/images/%s' % image_id)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'get_image': '!', 'delete_image': ''})
        resp = self.api_delete('/v2/images/%s' % image_id)
        self.assertEqual(204, resp.status_code)

    def test_image_upload(self):
        self.start_server()
        self._create_and_upload(expected_code=204)
        self.set_policy_rules({'add_image': '', 'get_image': '', 'upload_image': '!'})
        self._create_and_upload(expected_code=403)
        self.set_policy_rules({'add_image': '', 'get_image': '!', 'upload_image': '!'})
        self._create_and_upload(expected_code=404)
        self.set_policy_rules({'add_image': '', 'get_image': '!', 'upload_image': ''})
        self._create_and_upload(expected_code=204)

    def test_image_download(self):
        self.start_server()
        image_id = self._create_and_upload()
        path = '/v2/images/%s/file' % image_id
        response = self.api_get(path)
        self.assertEqual(200, response.status_code)
        self.assertEqual('IMAGEDATA', response.text)
        self.set_policy_rules({'get_image': '', 'download_image': '!'})
        response = self.api_get(path)
        self.assertEqual(403, response.status_code)
        self.set_policy_rules({'get_image': '!', 'download_image': '!'})
        response = self.api_get(path)
        self.assertEqual(404, response.status_code)
        self.set_policy_rules({'get_image': '!', 'download_image': ''})
        response = self.api_get(path)
        self.assertEqual(200, response.status_code)
        self.assertEqual('IMAGEDATA', response.text)

    def test_image_stage(self):
        self.start_server()
        self._create_and_stage(expected_code=204)
        self.set_policy_rules({'get_image': '!', 'modify_image': '', 'add_image': ''})
        self._create_and_stage(expected_code=204)
        self.set_policy_rules({'get_image': '', 'modify_image': '!', 'add_image': ''})
        self._create_and_stage(expected_code=403)
        self.set_policy_rules({'get_image': '!', 'modify_image': '!', 'add_image': ''})
        self._create_and_stage(expected_code=404)
        self.set_policy_rules({'get_image': '', 'modify_image': '!', 'add_image': '', 'add_member': ''})
        resp = self.api_post('/v2/images', json={'name': 'foo', 'container_format': 'bare', 'disk_format': 'raw', 'visibility': 'shared'})
        self.assertEqual(201, resp.status_code, resp.text)
        image = resp.json
        headers = self._headers({'X-Project-Id': 'fake-tenant-id', 'Content-Type': 'application/octet-stream'})
        resp = self.api_put('/v2/images/%s/stage' % image['id'], headers=headers, data=b'IMAGEDATA')
        self.assertEqual(404, resp.status_code)
        path = '/v2/images/%s/members' % image['id']
        data = {'member': uuids.random_member}
        response = self.api_post(path, json=data)
        member = response.json
        self.assertEqual(200, response.status_code)
        self.assertEqual(image['id'], member['image_id'])
        headers = self._headers({'X-Project-Id': uuids.random_member, 'X-Roles': 'member', 'Content-Type': 'application/octet-stream'})
        resp = self.api_put('/v2/images/%s/stage' % image['id'], headers=headers, data=b'IMAGEDATA')
        self.assertEqual(403, resp.status_code)

    def test_image_deactivate(self):
        self.start_server()
        image_id = self._create_and_upload()
        resp = self.api_post('/v2/images/%s/actions/deactivate' % image_id)
        self.assertEqual(204, resp.status_code)
        resp = self.api_get('/v2/images/%s' % image_id)
        self.assertEqual('deactivated', resp.json['status'])
        image_id = self._create_and_upload()
        self.set_policy_rules({'get_image': '', 'deactivate': '!'})
        resp = self.api_post('/v2/images/%s/actions/deactivate' % image_id)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'get_image': '!', 'deactivate': '!'})
        resp = self.api_post('/v2/images/%s/actions/deactivate' % image_id)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'get_image': '!', 'deactivate': ''})
        resp = self.api_post('/v2/images/%s/actions/deactivate' % image_id)
        self.assertEqual(204, resp.status_code)
        self.set_policy_rules({'get_image': '', 'modify_image': '', 'add_image': '', 'upload_image': '', 'add_member': '', 'deactivate': '', 'publicize_image': '', 'communitize_image': ''})
        headers = self._headers({'X-Project-Id': 'fake-project-id', 'X-Roles': 'member'})
        for visibility in ('community', 'shared', 'private', 'public'):
            image_id = self._create_and_upload(visibility=visibility)
            resp = self.api_post('/v2/images/%s/actions/deactivate' % image_id, headers=headers)
            if visibility == 'shared':
                self.assertEqual(404, resp.status_code)
                share_path = '/v2/images/%s/members' % image_id
                data = {'member': 'fake-project-id'}
                response = self.api_post(share_path, json=data)
                member = response.json
                self.assertEqual(200, response.status_code)
                self.assertEqual(image_id, member['image_id'])
                resp = self.api_post('/v2/images/%s/actions/deactivate' % image_id, headers=headers)
                self.assertEqual(403, resp.status_code)
            elif visibility == 'private':
                self.assertEqual(404, resp.status_code)
            else:
                self.assertEqual(403, resp.status_code)

    def test_image_reactivate(self):
        self.start_server()
        image_id = self._create_and_upload()
        resp = self.api_post('/v2/images/%s/actions/deactivate' % image_id)
        self.assertEqual(204, resp.status_code)
        resp = self.api_get('/v2/images/%s' % image_id)
        self.assertEqual('deactivated', resp.json['status'])
        resp = self.api_post('/v2/images/%s/actions/reactivate' % image_id)
        self.assertEqual(204, resp.status_code)
        resp = self.api_get('/v2/images/%s' % image_id)
        self.assertEqual('active', resp.json['status'])
        resp = self.api_post('/v2/images/%s/actions/deactivate' % image_id)
        self.assertEqual(204, resp.status_code)
        self.set_policy_rules({'get_image': '', 'reactivate': '!'})
        resp = self.api_post('/v2/images/%s/actions/reactivate' % image_id)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'get_image': '!', 'reactivate': '!'})
        resp = self.api_post('/v2/images/%s/actions/reactivate' % image_id)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'get_image': '!', 'reactivate': ''})
        resp = self.api_post('/v2/images/%s/actions/reactivate' % image_id)
        self.assertEqual(204, resp.status_code)
        self.set_policy_rules({'get_image': '', 'modify_image': '', 'add_image': '', 'upload_image': '', 'add_member': '', 'deactivate': '', 'reactivate': '', 'publicize_image': '', 'communitize_image': ''})
        headers = self._headers({'X-Project-Id': 'fake-project-id', 'X-Roles': 'member'})
        for visibility in ('public', 'community', 'shared', 'private'):
            image_id = self._create_and_upload(visibility=visibility)
            resp = self.api_post('/v2/images/%s/actions/deactivate' % image_id)
            self.assertEqual(204, resp.status_code)
            resp = self.api_post('/v2/images/%s/actions/reactivate' % image_id, headers=headers)
            if visibility == 'shared':
                self.assertEqual(404, resp.status_code)
                share_path = '/v2/images/%s/members' % image_id
                data = {'member': 'fake-project-id'}
                response = self.api_post(share_path, json=data)
                member = response.json
                self.assertEqual(200, response.status_code)
                self.assertEqual(image_id, member['image_id'])
                resp = self.api_post('/v2/images/%s/actions/reactivate' % image_id, headers=headers)
                self.assertEqual(403, resp.status_code)
            elif visibility == 'private':
                self.assertEqual(404, resp.status_code)
            else:
                self.assertEqual(403, resp.status_code)

    def test_delete_from_store(self):
        self.start_server()
        image_id = self._create_and_import(stores=['store1', 'store2', 'store3'])
        path = '/v2/stores/store1/%s' % image_id
        response = self.api_delete(path)
        self.assertEqual(204, response.status_code)
        self.set_policy_rules({'get_image': '', 'delete_image_location': '', 'get_image_location': '!'})
        path = '/v2/stores/store2/%s' % image_id
        response = self.api_delete(path)
        self.assertEqual(403, response.status_code)
        self.set_policy_rules({'get_image': '', 'delete_image_location': '!', 'get_image_location': ''})
        path = '/v2/stores/store2/%s' % image_id
        response = self.api_delete(path)
        self.assertEqual(403, response.status_code)
        self.set_policy_rules({'get_image': '!', 'delete_image_location': '!', 'get_image_location': '!'})
        path = '/v2/stores/store2/%s' % image_id
        response = self.api_delete(path)
        self.assertEqual(404, response.status_code)
        self.set_policy_rules({'get_image': '!', 'delete_image_location': '', 'get_image_location': ''})
        path = '/v2/stores/store2/%s' % image_id
        response = self.api_delete(path)
        self.assertEqual(204, response.status_code)
        self.set_policy_rules({'get_image': '', 'delete_image_location': '', 'get_image_location': ''})
        headers = self._headers({'X-Roles': 'member'})
        path = '/v2/stores/store2/%s' % image_id
        response = self.api_delete(path, headers=headers)
        self.assertEqual(403, response.status_code)

    def test_copy_image(self):
        self.start_server()
        image_id = self._create_and_import(stores=['store1'], visibility='public')
        self.set_policy_rules({'copy_image': 'role:admin', 'get_image': '', 'modify_image': ''})
        store_to_copy = ['store2']
        response = self._import_copy(image_id, store_to_copy)
        self.assertEqual(202, response.status_code)
        self._wait_for_import(image_id)
        self.assertEqual('success', self._get_latest_task(image_id)['status'])
        store_to_copy = ['store3']
        self.set_policy_rules({'copy_image': '!', 'get_image': '', 'modify_image': ''})
        response = self._import_copy(image_id, store_to_copy)
        self.assertEqual(403, response.status_code)
        self.set_policy_rules({'copy_image': 'role:admin', 'get_image': '', 'modify_image': ''})
        headers = self._headers({'X-Roles': 'member'})
        response = self._import_copy(image_id, store_to_copy, headers=headers)
        self.assertEqual(403, response.status_code)
        self.set_policy_rules({'copy_image': 'role:admin', 'get_image': '', 'modify_image': ''})
        headers = self._headers({'X-Roles': 'member', 'X-Project-Id': 'fake-project-id'})
        response = self._import_copy(image_id, store_to_copy, headers=headers)
        self.assertEqual(403, response.status_code)
        self.set_policy_rules({'copy_image': '!', 'get_image': '!', 'modify_image': ''})
        store_to_copy = ['store3']
        print(self.policy.rules.items())
        response = self._import_copy(image_id, store_to_copy)
        self.assertEqual(404, response.status_code)

    def test_import_glance_direct(self):
        self.start_server()
        image_id = self._create_and_stage(visibility='public')
        self.set_policy_rules({'get_image': '', 'communitize_image': '', 'add_image': '', 'modify_image': ''})
        store_to_import = ['store1']
        response = self._import_direct(image_id, store_to_import)
        self.assertEqual(202, response.status_code)
        self._wait_for_import(image_id)
        self.assertEqual('success', self._get_latest_task(image_id)['status'])
        image_id = self._create_and_stage(visibility='community')
        headers = self._headers({'X-Roles': 'member'})
        response = self._import_direct(image_id, store_to_import, headers=headers)
        self.assertEqual(202, response.status_code)
        self._wait_for_import(image_id)
        self.assertEqual('success', self._get_latest_task(image_id)['status'])
        image_id = self._create_and_stage(visibility='community')
        self.set_policy_rules({'get_image': '', 'modify_image': '!'})
        headers = self._headers({'X-Roles': 'member', 'X-Project-Id': 'fake-project-id'})
        response = self._import_direct(image_id, store_to_import, headers=headers)
        self.assertEqual(403, response.status_code)
        self.set_policy_rules({'get_image': '!', 'modify_image': '!'})
        headers = self._headers({'X-Roles': 'member', 'X-Project-Id': 'fake-project-id'})
        response = self._import_direct(image_id, store_to_import, headers=headers)
        self.assertEqual(404, response.status_code)

    def _test_image_ownership(self, headers, method):
        self.set_policy_rules({'get_image': '', 'add_image': '', 'publicize_image': '', 'communitize_image': '', 'add_member': ''})
        for visibility in ('community', 'public', 'shared'):
            path = '/v2/images'
            data = {'name': '%s-image' % visibility, 'visibility': visibility}
            response = self.api_post(path, json=data)
            image = response.json
            self.assertEqual(201, response.status_code)
            self.assertEqual(visibility, image['visibility'])
            if visibility == 'shared':
                path = '/v2/images/%s/members' % image['id']
                data = {'member': 'fake-project-id'}
                response = self.api_post(path, json=data)
                self.assertEqual(200, response.status_code)
            path = '/v2/images/%s/tags/Test_Tag_2' % image['id']
            response = self.api_request(method, path, headers=headers)
            self.assertEqual(403, response.status_code)

    def test_image_tag_update(self):
        self.start_server()
        image_id = self._create_and_upload()
        path = '/v2/images/%s/tags/Test_Tag' % image_id
        response = self.api_put(path)
        self.assertEqual(204, response.status_code)
        path = '/v2/images/%s' % image_id
        response = self.api_get(path)
        image = response.json
        self.assertEqual(['Test_Tag'], image['tags'])
        self.set_policy_rules({'get_image': '!', 'modify_image': '!'})
        path = '/v2/images/%s/tags/Test_Tag_2' % image_id
        response = self.api_put(path)
        self.assertEqual(404, response.status_code)
        self.set_policy_rules({'get_image': '', 'modify_image': '!'})
        path = '/v2/images/%s/tags/Test_Tag_2' % image_id
        response = self.api_put(path)
        self.assertEqual(403, response.status_code)
        self.set_policy_rules({'get_image': '', 'modify_image': ''})
        headers = self._headers({'X-Project-Id': 'fake-project-id', 'X-Roles': 'member'})
        path = '/v2/images/%s/tags/Test_Tag_2' % image_id
        response = self.api_put(path, headers=headers)
        self.assertEqual(404, response.status_code)
        self._test_image_ownership(headers, 'PUT')

    def test_image_tag_delete(self):
        self.start_server()
        image_id = self._create_and_upload()
        path = '/v2/images/%s/tags/Test_Tag_1' % image_id
        response = self.api_put(path)
        self.assertEqual(204, response.status_code)
        path = '/v2/images/%s/tags/Test_Tag_2' % image_id
        response = self.api_put(path)
        self.assertEqual(204, response.status_code)
        path = '/v2/images/%s' % image_id
        response = self.api_get(path)
        image = response.json
        self.assertItemsEqual(['Test_Tag_1', 'Test_Tag_2'], image['tags'])
        path = '/v2/images/%s/tags/Test_Tag_1' % image_id
        response = self.api_delete(path)
        self.assertEqual(204, response.status_code)
        path = '/v2/images/%s' % image_id
        response = self.api_get(path)
        image = response.json
        self.assertNotIn('Test_Tag_1', image['tags'])
        self.set_policy_rules({'get_image': '!', 'modify_image': '!'})
        path = '/v2/images/%s/tags/Test_Tag_2' % image_id
        response = self.api_delete(path)
        self.assertEqual(404, response.status_code)
        self.set_policy_rules({'get_image': '', 'modify_image': '!'})
        path = '/v2/images/%s/tags/Test_Tag_2' % image_id
        response = self.api_delete(path)
        self.assertEqual(403, response.status_code)
        self.set_policy_rules({'get_image': '', 'modify_image': ''})
        headers = self._headers({'X-Project-Id': 'fake-project-id', 'X-Roles': 'member'})
        path = '/v2/images/%s/tags/Test_Tag_2' % image_id
        response = self.api_delete(path, headers=headers)
        self.assertEqual(404, response.status_code)
        self._test_image_ownership(headers, 'DELETE')

    def test_get_task_info(self):
        self.start_server()
        image_id = self._create_and_import(stores=['store1'], visibility='public')
        path = '/v2/images/%s/tasks' % image_id
        response = self.api_get(path)
        self.assertEqual(200, response.status_code)
        self.set_policy_rules({'get_image': '!'})
        path = '/v2/images/%s/tasks' % image_id
        response = self.api_get(path)
        self.assertEqual(404, response.status_code)