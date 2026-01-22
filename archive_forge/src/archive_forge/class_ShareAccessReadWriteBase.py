import ast
import ddt
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient import api_versions
from manilaclient import config
from manilaclient.tests.functional import base
@ddt.ddt
class ShareAccessReadWriteBase(base.BaseTestCase):
    protocol = None
    access_level = None

    def setUp(self):
        super(ShareAccessReadWriteBase, self).setUp()
        if self.protocol not in CONF.enable_protocols:
            message = '%s tests are disabled.' % self.protocol
            raise self.skipException(message)
        if self.access_level not in CONF.access_levels_mapping.get(self.protocol, '').split(' '):
            raise self.skipException('%(level)s tests for %(protocol)s share access are disabled.' % {'level': self.access_level, 'protocol': self.protocol})
        self.access_types = CONF.access_types_mapping.get(self.protocol, '').split(' ')
        if not self.access_types:
            raise self.skipException('No access levels were provided for %s share access tests.' % self.protocol)
        self.share = self.create_share(share_protocol=self.protocol, public=True)
        self.share_id = self.share['id']
        int_range = range(20, 50)
        self.access_to = {'ip': ['99.88.77.%d' % i for i in int_range], 'user': ['foo_user_%d' % i for i in int_range], 'cert': ['tenant_%d.example.com' % i for i in int_range], 'ipv6': ['2001:db8::%d' % i for i in int_range]}

    def _test_create_list_access_rule_for_share(self, microversion, metadata=None):
        access_type = self.access_types[0]
        access = self.user_client.access_allow(self.share['id'], access_type, self.access_to[access_type].pop(), self.access_level, metadata=metadata, microversion=microversion)
        return access

    @ddt.data(*set(['1.0', '2.0', '2.6', '2.7', '2.21', '2.33', '2.44', '2.45', api_versions.MAX_VERSION]))
    def test_create_list_access_rule_for_share(self, microversion):
        self.skip_if_microversion_not_supported(microversion)
        access = self._test_create_list_access_rule_for_share(microversion=microversion)
        access_list = self.user_client.list_access(self.share['id'], microversion=microversion)
        self.assertTrue(any([item for item in access_list if access['id'] == item['id']]))
        self.assertTrue(any((a['access_type'] is not None for a in access_list)))
        self.assertTrue(any((a['access_to'] is not None for a in access_list)))
        self.assertTrue(any((a['access_level'] is not None for a in access_list)))
        if api_versions.APIVersion(microversion) >= api_versions.APIVersion('2.33'):
            self.assertTrue(all((all((key in access for key in ('access_key', 'created_at', 'updated_at'))) for access in access_list)))
        elif api_versions.APIVersion(microversion) >= api_versions.APIVersion('2.21'):
            self.assertTrue(all(('access_key' in a for a in access_list)))
        else:
            self.assertTrue(all(('access_key' not in a for a in access_list)))

    @ddt.data('1.0', '2.0', '2.6', '2.7')
    def test_create_list_access_rule_for_share_select_column(self, microversion):
        self.skip_if_microversion_not_supported(microversion)
        self._test_create_list_access_rule_for_share(microversion=microversion)
        access_list = self.user_client.list_access(self.share['id'], columns='access_type,access_to', microversion=microversion)
        self.assertTrue(any((a['Access_Type'] is not None for a in access_list)))
        self.assertTrue(any((a['Access_To'] is not None for a in access_list)))
        self.assertTrue(all(('Access_Level' not in a for a in access_list)))
        self.assertTrue(all(('access_level' not in a for a in access_list)))

    def _create_delete_access_rule(self, share_id, access_type, access_to, microversion=None):
        self.skip_if_microversion_not_supported(microversion)
        if access_type not in self.access_types:
            raise self.skipException("'%(access_type)s' access rules is disabled for protocol '%(protocol)s'." % {'access_type': access_type, 'protocol': self.protocol})
        access = self.user_client.access_allow(share_id, access_type, access_to, self.access_level, microversion=microversion)
        self.assertEqual(share_id, access.get('share_id'))
        self.assertEqual(access_type, access.get('access_type'))
        self.assertEqual(access_to.replace('\\\\', '\\'), access.get('access_to'))
        self.assertEqual(self.access_level, access.get('access_level'))
        if api_versions.APIVersion(microversion) >= api_versions.APIVersion('2.33'):
            self.assertIn('access_key', access)
            self.assertIn('created_at', access)
            self.assertIn('updated_at', access)
        elif api_versions.APIVersion(microversion) >= api_versions.APIVersion('2.21'):
            self.assertIn('access_key', access)
        else:
            self.assertNotIn('access_key', access)
        self.user_client.wait_for_access_rule_status(share_id, access['id'])
        self.user_client.access_deny(share_id, access['id'])
        self.user_client.wait_for_access_rule_deletion(share_id, access['id'])
        self.assertRaises(tempest_lib_exc.NotFound, self.user_client.get_access, share_id, access['id'])

    @ddt.data(*set(['2.45', api_versions.MAX_VERSION]))
    def test_create_list_access_rule_with_metadata(self, microversion):
        self.skip_if_microversion_not_supported(microversion)
        md1 = {'key1': 'value1', 'key2': 'value2'}
        md2 = {'key3': 'value3', 'key4': 'value4'}
        self._test_create_list_access_rule_for_share(metadata=md1, microversion=microversion)
        access = self._test_create_list_access_rule_for_share(metadata=md2, microversion=microversion)
        access_list = self.user_client.list_access(self.share['id'], metadata={'key4': 'value4'}, microversion=microversion)
        self.assertEqual(1, len(access_list))
        get_access = self.user_client.access_show(access_list[0]['id'], microversion=microversion)
        metadata = ast.literal_eval(get_access['metadata'])
        self.assertEqual(2, len(metadata))
        self.assertIn('key3', metadata)
        self.assertIn('key4', metadata)
        self.assertEqual(md2['key3'], metadata['key3'])
        self.assertEqual(md2['key4'], metadata['key4'])
        self.assertEqual(access['id'], access_list[0]['id'])
        self.user_client.access_deny(access['share_id'], access['id'])
        self.user_client.wait_for_access_rule_deletion(access['share_id'], access['id'])

    @ddt.data(*set(['2.45', api_versions.MAX_VERSION]))
    def test_create_update_show_access_rule_with_metadata(self, microversion):
        self.skip_if_microversion_not_supported(microversion)
        md1 = {'key1': 'value1', 'key2': 'value2'}
        md2 = {'key3': 'value3', 'key2': 'value4'}
        access = self._test_create_list_access_rule_for_share(metadata=md1, microversion=microversion)
        get_access = self.user_client.access_show(access['id'], microversion=microversion)
        self.assertEqual(access['id'], get_access['id'])
        self.assertEqual(md1, ast.literal_eval(get_access['metadata']))
        self.user_client.access_set_metadata(access['id'], metadata=md2, microversion=microversion)
        get_access = self.user_client.access_show(access['id'], microversion=microversion)
        self.assertEqual({'key1': 'value1', 'key2': 'value4', 'key3': 'value3'}, ast.literal_eval(get_access['metadata']))
        self.assertEqual(access['id'], get_access['id'])

    @ddt.data(*set(['2.45', api_versions.MAX_VERSION]))
    def test_delete_access_rule_metadata(self, microversion):
        self.skip_if_microversion_not_supported(microversion)
        md = {'key1': 'value1', 'key2': 'value2'}
        access = self._test_create_list_access_rule_for_share(metadata=md, microversion=microversion)
        get_access = self.user_client.access_show(access['id'], microversion=microversion)
        self.assertEqual(access['id'], get_access['id'])
        self.assertEqual(md, ast.literal_eval(get_access['metadata']))
        self.user_client.access_unset_metadata(access['id'], keys=['key1', 'key2'], microversion=microversion)
        get_access = self.user_client.access_show(access['id'], microversion=microversion)
        self.assertEqual({}, ast.literal_eval(get_access['metadata']))
        self.assertEqual(access['id'], get_access['id'])

    @ddt.data('1.0', '2.0', '2.6', '2.7', '2.21', '2.33')
    def test_create_delete_ip_access_rule(self, microversion):
        self._create_delete_access_rule(self.share_id, 'ip', self.access_to['ip'].pop(), microversion)

    @ddt.data('1.0', '2.0', '2.6', '2.7', '2.21', '2.33')
    def test_create_delete_user_access_rule(self, microversion):
        self._create_delete_access_rule(self.share_id, 'user', CONF.username_for_user_rules, microversion)

    @ddt.data('1.0', '2.0', '2.6', '2.7', '2.21', '2.33')
    def test_create_delete_cert_access_rule(self, microversion):
        self._create_delete_access_rule(self.share_id, 'cert', self.access_to['cert'].pop(), microversion)

    @ddt.data('2.38', api_versions.MAX_VERSION)
    def test_create_delete_ipv6_access_rule(self, microversion):
        self._create_delete_access_rule(self.share_id, 'ip', self.access_to['ipv6'].pop(), microversion)