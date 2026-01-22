import datetime
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.image_members
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
class TestImageMembersController(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImageMembersController, self).setUp()
        self.db = unit_test_utils.FakeDB(initialize=False)
        self.store = unit_test_utils.FakeStoreAPI()
        self.policy = unit_test_utils.FakePolicyEnforcer()
        self.notifier = unit_test_utils.FakeNotifier()
        self._create_images()
        self._create_image_members()
        self.controller = glance.api.v2.image_members.ImageMembersController(self.db, self.policy, self.notifier, self.store)
        glance_store.register_opts(CONF)
        self.config(default_store='filesystem', filesystem_store_datadir=self.test_dir, group='glance_store')
        glance_store.create_stores()

    def _create_images(self):
        self.images = [_db_fixture(UUID1, owner=TENANT1, name='1', size=256, visibility='public', locations=[{'url': '%s/%s' % (BASE_URI, UUID1), 'metadata': {}, 'status': 'active'}]), _db_fixture(UUID2, owner=TENANT1, name='2', size=512), _db_fixture(UUID3, owner=TENANT3, name='3', size=512), _db_fixture(UUID4, owner=TENANT4, name='4', size=1024), _db_fixture(UUID5, owner=TENANT1, name='5', size=1024)]
        [self.db.image_create(None, image) for image in self.images]
        self.db.image_tag_set_all(None, UUID1, ['ping', 'pong'])

    def _create_image_members(self):
        self.image_members = [_db_image_member_fixture(UUID2, TENANT4), _db_image_member_fixture(UUID3, TENANT4), _db_image_member_fixture(UUID3, TENANT2), _db_image_member_fixture(UUID4, TENANT1)]
        [self.db.image_member_create(None, image_member) for image_member in self.image_members]

    def test_index(self):
        request = unit_test_utils.get_fake_request()
        output = self.controller.index(request, UUID2)
        self.assertEqual(1, len(output['members']))
        actual = set([image_member.member_id for image_member in output['members']])
        expected = set([TENANT4])
        self.assertEqual(expected, actual)

    def test_index_no_members(self):
        request = unit_test_utils.get_fake_request()
        output = self.controller.index(request, UUID5)
        self.assertEqual(0, len(output['members']))
        self.assertEqual({'members': []}, output)

    def test_index_member_view(self):
        request = unit_test_utils.get_fake_request(tenant=TENANT4)
        output = self.controller.index(request, UUID3)
        self.assertEqual(1, len(output['members']))
        actual = set([image_member.member_id for image_member in output['members']])
        expected = set([TENANT4])
        self.assertEqual(expected, actual)

    def test_index_private_image(self):
        request = unit_test_utils.get_fake_request(tenant=TENANT2)
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.index, request, UUID5)

    def test_index_public_image(self):
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.index, request, UUID1)

    def test_index_private_image_visible_members_admin(self):
        request = unit_test_utils.get_fake_request(is_admin=True)
        output = self.controller.index(request, UUID4)
        self.assertEqual(1, len(output['members']))
        actual = set([image_member.member_id for image_member in output['members']])
        expected = set([TENANT1])
        self.assertEqual(expected, actual)

    def test_index_allowed_by_get_members_policy(self):
        rules = {'get_members': True}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        output = self.controller.index(request, UUID2)
        self.assertEqual(1, len(output['members']))

    def test_index_forbidden_by_get_members_policy(self):
        rules = {'get_members': False}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.index, request, image_id=UUID2)

    def test_show(self):
        request = unit_test_utils.get_fake_request(tenant=TENANT1)
        output = self.controller.show(request, UUID2, TENANT4)
        expected = self.image_members[0]
        self.assertEqual(expected['image_id'], output.image_id)
        self.assertEqual(expected['member'], output.member_id)
        self.assertEqual(expected['status'], output.status)

    def test_show_by_member(self):
        request = unit_test_utils.get_fake_request(tenant=TENANT4)
        output = self.controller.show(request, UUID2, TENANT4)
        expected = self.image_members[0]
        self.assertEqual(expected['image_id'], output.image_id)
        self.assertEqual(expected['member'], output.member_id)
        self.assertEqual(expected['status'], output.status)

    def test_show_forbidden(self):
        request = unit_test_utils.get_fake_request(tenant=TENANT2)
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.show, request, UUID2, TENANT4)

    def test_show_not_found(self):
        request = unit_test_utils.get_fake_request(tenant=TENANT2)
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.show, request, UUID3, TENANT4)

    def test_create(self):
        request = unit_test_utils.get_fake_request()
        image_id = UUID2
        member_id = TENANT3
        output = self.controller.create(request, image_id=image_id, member_id=member_id)
        self.assertEqual(UUID2, output.image_id)
        self.assertEqual(TENANT3, output.member_id)

    def test_create_allowed_by_add_policy(self):
        rules = {'add_member': True}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        output = self.controller.create(request, image_id=UUID2, member_id=TENANT3)
        self.assertEqual(UUID2, output.image_id)
        self.assertEqual(TENANT3, output.member_id)

    def test_create_forbidden_by_add_policy(self):
        rules = {'add_member': False}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.create, request, image_id=UUID2, member_id=TENANT3)

    def test_create_duplicate_member(self):
        request = unit_test_utils.get_fake_request()
        image_id = UUID2
        member_id = TENANT3
        output = self.controller.create(request, image_id=image_id, member_id=member_id)
        self.assertEqual(UUID2, output.image_id)
        self.assertEqual(TENANT3, output.member_id)
        self.assertRaises(webob.exc.HTTPConflict, self.controller.create, request, image_id=image_id, member_id=member_id)

    def test_create_overlimit(self):
        self.config(image_member_quota=0)
        request = unit_test_utils.get_fake_request()
        image_id = UUID2
        member_id = TENANT3
        self.assertRaises(webob.exc.HTTPRequestEntityTooLarge, self.controller.create, request, image_id=image_id, member_id=member_id)

    def test_create_unlimited(self):
        self.config(image_member_quota=-1)
        request = unit_test_utils.get_fake_request()
        image_id = UUID2
        member_id = TENANT3
        output = self.controller.create(request, image_id=image_id, member_id=member_id)
        self.assertEqual(UUID2, output.image_id)
        self.assertEqual(TENANT3, output.member_id)

    def test_member_create_raises_bad_request_for_unicode_value(self):
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.create, request, image_id=UUID5, member_id='🚓')

    def test_update_done_by_member(self):
        request = unit_test_utils.get_fake_request(tenant=TENANT4)
        image_id = UUID2
        member_id = TENANT4
        output = self.controller.update(request, image_id=image_id, member_id=member_id, status='accepted')
        self.assertEqual(UUID2, output.image_id)
        self.assertEqual(TENANT4, output.member_id)
        self.assertEqual('accepted', output.status)

    def test_update_done_by_member_forbidden_by_policy(self):
        rules = {'modify_member': False}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request(tenant=TENANT4)
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, image_id=UUID2, member_id=TENANT4, status='accepted')

    def test_update_done_by_member_allowed_by_policy(self):
        rules = {'modify_member': True}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request(tenant=TENANT4)
        output = self.controller.update(request, image_id=UUID2, member_id=TENANT4, status='accepted')
        self.assertEqual(UUID2, output.image_id)
        self.assertEqual(TENANT4, output.member_id)
        self.assertEqual('accepted', output.status)

    def test_update_done_by_owner(self):
        enforcer = unit_test_utils.enforcer_from_rules({'get_image': '', 'modify_image': "'{0}':%(owner)s".format(TENANT1)})
        self.controller.policy = enforcer
        request = unit_test_utils.get_fake_request(tenant=TENANT1)
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, UUID2, TENANT4, status='accepted')

    def test_update_non_existent_image(self):
        request = unit_test_utils.get_fake_request(tenant=TENANT1)
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.update, request, '123', TENANT4, status='accepted')

    def test_update_invalid_status(self):
        request = unit_test_utils.get_fake_request(tenant=TENANT4)
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.update, request, UUID2, TENANT4, status='accept')

    def test_create_private_image(self):
        enforcer = unit_test_utils.enforcer_from_rules({'get_image': ''})
        self.controller.policy = enforcer
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.create, request, UUID4, TENANT2)

    def test_create_public_image(self):
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.create, request, UUID1, TENANT2)

    def test_create_image_does_not_exist(self):
        request = unit_test_utils.get_fake_request()
        image_id = 'fake-image-id'
        member_id = TENANT3
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.create, request, image_id=image_id, member_id=member_id)

    def test_delete(self):
        request = unit_test_utils.get_fake_request()
        member_id = TENANT4
        image_id = UUID2
        res = self.controller.delete(request, image_id, member_id)
        self.assertEqual(b'', res.body)
        self.assertEqual(http.NO_CONTENT, res.status_code)
        found_member = self.db.image_member_find(request.context, image_id=image_id, member=member_id)
        self.assertEqual([], found_member)

    def test_delete_by_member(self):
        enforcer = unit_test_utils.enforcer_from_rules({'get_image': '', 'delete_member': "'{0}':%(owner)s".format(TENANT4), 'get_members': '', 'get_member': ''})
        request = unit_test_utils.get_fake_request(tenant=TENANT4)
        self.controller.policy = enforcer
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.delete, request, UUID2, TENANT4)
        request = unit_test_utils.get_fake_request()
        output = self.controller.index(request, UUID2)
        self.assertEqual(1, len(output['members']))
        actual = set([image_member.member_id for image_member in output['members']])
        expected = set([TENANT4])
        self.assertEqual(expected, actual)

    def test_delete_allowed_by_policies(self):
        rules = {'get_member': True, 'delete_member': True}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request(tenant=TENANT1)
        output = self.controller.delete(request, image_id=UUID2, member_id=TENANT4)
        request = unit_test_utils.get_fake_request()
        output = self.controller.index(request, UUID2)
        self.assertEqual(0, len(output['members']))

    def test_delete_forbidden_by_get_member_policy(self):
        rules = {'get_member': False}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request(tenant=TENANT1)
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.delete, request, UUID2, TENANT4)

    def test_delete_forbidden_by_delete_member_policy(self):
        rules = {'delete_member': False}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request(tenant=TENANT1)
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.delete, request, UUID2, TENANT4)

    def test_delete_private_image(self):
        enforcer = unit_test_utils.enforcer_from_rules({'get_image': '', 'delete_member': "'{0}':%(owner)s".format(TENANT1), 'get_member': ''})
        self.controller.policy = enforcer
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.delete, request, UUID4, TENANT1)

    def test_delete_public_image(self):
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.delete, request, UUID1, TENANT1)

    def test_delete_image_does_not_exist(self):
        request = unit_test_utils.get_fake_request()
        member_id = TENANT2
        image_id = 'fake-image-id'
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete, request, image_id, member_id)

    def test_delete_member_does_not_exist(self):
        request = unit_test_utils.get_fake_request()
        member_id = 'fake-member-id'
        image_id = UUID2
        found_member = self.db.image_member_find(request.context, image_id=image_id, member=member_id)
        self.assertEqual([], found_member)
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete, request, image_id, member_id)