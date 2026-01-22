import testtools
from glanceclient.tests import utils
import glanceclient.v1.image_members
import glanceclient.v1.images
class ImageMemberManagerTest(testtools.TestCase):

    def setUp(self):
        super(ImageMemberManagerTest, self).setUp()
        self.api = utils.FakeAPI(fixtures)
        self.mgr = glanceclient.v1.image_members.ImageMemberManager(self.api)
        self.image = glanceclient.v1.images.Image(self.api, {'id': '1'}, True)

    def test_list_by_image(self):
        members = self.mgr.list(image=self.image)
        expect = [('GET', '/v1/images/1/members', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(members))
        self.assertEqual('1', members[0].member_id)
        self.assertEqual('1', members[0].image_id)
        self.assertEqual(False, members[0].can_share)

    def test_list_by_member(self):
        resource_class = glanceclient.v1.image_members.ImageMember
        member = resource_class(self.api, {'member_id': '1'}, True)
        self.mgr.list(member=member)
        expect = [('GET', '/v1/shared-images/1', {}, None)]
        self.assertEqual(expect, self.api.calls)

    def test_get(self):
        member = self.mgr.get(self.image, '1')
        expect = [('GET', '/v1/images/1/members/1', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual('1', member.member_id)
        self.assertEqual('1', member.image_id)
        self.assertEqual(False, member.can_share)

    def test_delete(self):
        self.mgr.delete('1', '1')
        expect = [('DELETE', '/v1/images/1/members/1', {}, None)]
        self.assertEqual(expect, self.api.calls)

    def test_create(self):
        self.mgr.create(self.image, '1', can_share=True)
        expect_body = {'member': {'can_share': True}}
        expect = [('PUT', '/v1/images/1/members/1', {}, sorted(expect_body.items()))]
        self.assertEqual(expect, self.api.calls)

    def test_replace(self):
        body = [{'member_id': '2', 'can_share': False}, {'member_id': '3'}]
        self.mgr.replace(self.image, body)
        expect = [('PUT', '/v1/images/1/members', {}, sorted({'memberships': body}.items()))]
        self.assertEqual(expect, self.api.calls)

    def test_replace_objects(self):
        body = [glanceclient.v1.image_members.ImageMember(self.mgr, {'member_id': '2', 'can_share': False}, True), glanceclient.v1.image_members.ImageMember(self.mgr, {'member_id': '3', 'can_share': True}, True)]
        self.mgr.replace(self.image, body)
        expect_body = {'memberships': [{'member_id': '2', 'can_share': False}, {'member_id': '3', 'can_share': True}]}
        expect = [('PUT', '/v1/images/1/members', {}, sorted(expect_body.items()))]
        self.assertEqual(expect, self.api.calls)