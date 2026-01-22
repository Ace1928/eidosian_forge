from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
class ActionIsolationCLITests(base_v2.MistralClientTestBase):

    def test_actions_name_uniqueness(self):
        self.action_create(self.act_def)
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'action-create', params='{0}'.format(self.act_def))
        self.action_create(self.act_def, admin=False)
        self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'action-create', params='{0}'.format(self.act_def))

    def test_action_isolation(self):
        act = self.action_create(self.act_def)
        acts = self.mistral_admin('action-list')
        self.assertIn(act[0]['Name'], [a['Name'] for a in acts])
        alt_acts = self.mistral_alt_user('action-list')
        self.assertNotIn(act[0]['Name'], [a['Name'] for a in alt_acts])

    def test_get_action_from_another_tenant(self):
        act = self.action_create(self.act_def)
        self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'action-get', params=act[0]['Name'])

    def test_delete_action_from_another_tenant(self):
        act = self.action_create(self.act_def)
        self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'action-delete', params=act[0]['Name'])

    def test_create_public_action(self):
        act = self.action_create(self.act_def, scope='public')
        same_act = self.mistral_alt_user('action-get', params=act[0]['Name'])
        self.assertEqual(act[0]['Name'], self.get_field_value(same_act, 'Name'))