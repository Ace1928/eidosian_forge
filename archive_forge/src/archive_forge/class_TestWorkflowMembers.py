import copy
from mistralclient.tests.unit.v2 import base
class TestWorkflowMembers(base.BaseClientV2Test):

    def test_create(self):
        self.requests_mock.post(self.TEST_URL + WORKFLOW_MEMBERS_URL, json=MEMBER, status_code=201)
        mb = self.members.create(MEMBER['resource_id'], MEMBER['resource_type'], MEMBER['member_id'])
        self.assertIsNotNone(mb)
        self.assertDictEqual({'member_id': MEMBER['member_id']}, self.requests_mock.last_request.json())

    def test_update(self):
        updated_member = copy.copy(MEMBER)
        updated_member['status'] = 'accepted'
        self.requests_mock.put(self.TEST_URL + WORKFLOW_MEMBER_URL, json=updated_member)
        mb = self.members.update(MEMBER['resource_id'], MEMBER['resource_type'], MEMBER['member_id'])
        self.assertIsNotNone(mb)
        self.assertDictEqual({'status': 'accepted'}, self.requests_mock.last_request.json())

    def test_list(self):
        self.requests_mock.get(self.TEST_URL + WORKFLOW_MEMBERS_URL, json={'members': [MEMBER]})
        mbs = self.members.list(MEMBER['resource_id'], MEMBER['resource_type'])
        self.assertEqual(1, len(mbs))

    def test_get(self):
        self.requests_mock.get(self.TEST_URL + WORKFLOW_MEMBER_URL, json=MEMBER)
        mb = self.members.get(MEMBER['resource_id'], MEMBER['resource_type'], MEMBER['member_id'])
        self.assertIsNotNone(mb)

    def test_delete(self):
        self.requests_mock.delete(self.TEST_URL + WORKFLOW_MEMBER_URL, status_code=204)
        self.members.delete(MEMBER['resource_id'], MEMBER['resource_type'], MEMBER['member_id'])