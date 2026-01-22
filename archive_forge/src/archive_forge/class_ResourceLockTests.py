from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as lib_exc
from manilaclient.tests.functional.osc import base
from manilaclient.tests.functional import utils
@utils.skip_if_microversion_not_supported('2.82')
class ResourceLockTests(base.OSCClientTestBase):
    """Lock CLI test cases"""

    def setUp(self):
        super(ResourceLockTests, self).setUp()
        self.share_type = self.create_share_type(name=data_utils.rand_name('lock_tests_type'))
        self.share = self.create_share(share_type=self.share_type['id'])

    def test_lock_create_show_use_delete(self):
        """Create a deletion lock on share, view it, try it and remove."""
        lock = self.create_resource_lock(self.share['id'], lock_reason='tigers rule', client=self.user_client, add_cleanup=False)
        client_user_id = self.openstack('token issue -c user_id -f value', client=self.user_client).strip()
        client_project_id = self.openstack('token issue -c project_id -f value', client=self.user_client).strip()
        self.assertEqual(self.share['id'], lock['resource_id'])
        self.assertEqual('delete', lock['resource_action'])
        self.assertEqual(client_user_id, lock['user_id'])
        self.assertEqual(client_project_id, lock['project_id'])
        self.assertEqual('user', lock['lock_context'])
        self.assertEqual('tigers rule', lock['lock_reason'])
        lock_show = self.dict_result('share', f'lock show {lock['id']}')
        self.assertEqual(lock['id'], lock_show['ID'])
        self.assertEqual(lock['lock_context'], lock_show['Lock Context'])
        self.assertRaises(lib_exc.CommandFailed, self.openstack, f'share delete {self.share['id']}')
        self.openstack(f'share lock delete {lock['id']}', client=self.user_client)
        self.assertRaises(lib_exc.CommandFailed, self.openstack, f'share lock show {lock['id']}')

    def test_lock_list_filter_paginate(self):
        lock_1 = self.create_resource_lock(self.share['id'], lock_reason='tigers rule', client=self.user_client)
        lock_2 = self.create_resource_lock(self.share['id'], lock_reason='tigers still rule', client=self.user_client)
        lock_3 = self.create_resource_lock(self.share['id'], lock_reason='admins rule', client=self.admin_client)
        locks = self.listing_result('share', f'lock list --resource {self.share['id']}')
        self.assertEqual(3, len(locks))
        self.assertEqual(sorted(LOCK_SUMMARY_ATTRIBUTES), sorted(locks[0].keys()))
        locks = self.listing_result('share', f'lock list --lock-context user  --resource {self.share['id']}')
        self.assertEqual(2, len(locks))
        self.assertNotIn(lock_3['id'], [lock['ID'] for lock in locks])
        locks = self.listing_result('share', f'lock list --lock-context user --resource {self.share['id']} --sort-key created_at  --sort-dir desc  --limit 1')
        self.assertEqual(1, len(locks))
        self.assertIn(lock_2['id'], [lock['ID'] for lock in locks])
        self.assertNotIn(lock_1['id'], [lock['ID'] for lock in locks])
        self.assertNotIn(lock_3['id'], [lock['ID'] for lock in locks])

    def test_lock_set_unset_lock_reason(self):
        lock = self.create_resource_lock(self.share['id'], client=self.user_client)
        self.assertEqual('None', lock['lock_reason'])
        self.openstack(f"share lock set --lock-reason 'updated reason' {lock['id']}")
        lock_show = self.dict_result('share', f'lock show {lock['id']}')
        self.assertEqual('updated reason', lock_show['Lock Reason'])
        self.openstack(f'share lock unset --lock-reason {lock['id']}')
        lock_show = self.dict_result('share', f'lock show {lock['id']}')
        self.assertEqual('None', lock_show['Lock Reason'])

    def test_lock_restrictions(self):
        """A user can't update or delete a lock created by another user."""
        lock = self.create_resource_lock(self.share['id'], client=self.admin_client, add_cleanup=False)
        self.assertEqual('admin', lock['lock_context'])
        self.assertRaises(lib_exc.CommandFailed, self.openstack, f"share lock set {lock['id']} --reason 'i cannot do this'", client=self.user_client)
        self.assertRaises(lib_exc.CommandFailed, self.openstack, f'share lock unset {lock['id']} --reason', client=self.user_client)
        self.assertRaises(lib_exc.CommandFailed, self.openstack, f'share lock delete {lock['id']} ', client=self.user_client)
        self.openstack(f'share lock set --lock-reason "I can do this" {lock['id']}', client=self.admin_client)
        self.openstack(f'share lock delete {lock['id']}', client=self.admin_client)