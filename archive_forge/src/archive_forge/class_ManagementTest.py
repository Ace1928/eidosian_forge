import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import management
class ManagementTest(testtools.TestCase):

    def setUp(self):
        super(ManagementTest, self).setUp()
        self.orig__init = management.Management.__init__
        management.Management.__init__ = mock.Mock(return_value=None)
        self.management = management.Management()
        self.management.api = mock.Mock()
        self.management.api.client = mock.Mock()
        self.orig_hist__init = management.RootHistory.__init__
        self.orig_base_getid = base.getid
        base.getid = mock.Mock(return_value='instance1')

    def tearDown(self):
        super(ManagementTest, self).tearDown()
        management.Management.__init__ = self.orig__init
        management.RootHistory.__init__ = self.orig_hist__init
        base.getid = self.orig_base_getid

    def test_show(self):

        def side_effect_func(path, instance):
            return (path, instance)
        self.management._get = mock.Mock(side_effect=side_effect_func)
        p, i = self.management.show(1)
        self.assertEqual(('/mgmt/instances/instance1', 'instance'), (p, i))

    def test_list(self):
        page_mock = mock.Mock()
        self.management._paginated = page_mock
        self.management.list(deleted=True)
        page_mock.assert_called_with('/mgmt/instances', 'instances', None, None, query_strings={'deleted': True})
        self.management.list(deleted=False, limit=10, marker='foo')
        page_mock.assert_called_with('/mgmt/instances', 'instances', 10, 'foo', query_strings={'deleted': False})

    def test_index(self):
        """index() is just wrapper for list()"""
        page_mock = mock.Mock()
        self.management._paginated = page_mock
        self.management.index(deleted=True)
        page_mock.assert_called_with('/mgmt/instances', 'instances', None, None, query_strings={'deleted': True})
        self.management.index(deleted=False, limit=10, marker='foo')
        page_mock.assert_called_with('/mgmt/instances', 'instances', 10, 'foo', query_strings={'deleted': False})

    def test_root_enabled_history(self):
        self.management.api.client.get = mock.Mock(return_value=('resp', None))
        self.assertRaises(Exception, self.management.root_enabled_history, 'instance')
        body = {'root_history': 'rh'}
        self.management.api.client.get = mock.Mock(return_value=('resp', body))
        management.RootHistory.__init__ = mock.Mock(return_value=None)
        rh = self.management.root_enabled_history('instance')
        self.assertIsInstance(rh, management.RootHistory)

    def test__action(self):
        resp = mock.Mock()
        self.management.api.client.post = mock.Mock(return_value=(resp, 'body'))
        resp.status_code = 200
        self.management._action(1, 'body')
        self.assertEqual(1, self.management.api.client.post.call_count)
        resp.status_code = 400
        self.assertRaises(Exception, self.management._action, 1, 'body')
        self.assertEqual(2, self.management.api.client.post.call_count)

    def _mock_action(self):
        self.body_ = ''

        def side_effect_func(instance_id, body):
            self.body_ = body
        self.management._action = mock.Mock(side_effect=side_effect_func)

    def test_stop(self):
        self._mock_action()
        self.management.stop(1)
        self.assertEqual(1, self.management._action.call_count)
        self.assertEqual({'stop': {}}, self.body_)

    def test_reboot(self):
        self._mock_action()
        self.management.reboot(1)
        self.assertEqual(1, self.management._action.call_count)
        self.assertEqual({'reboot': {}}, self.body_)

    def test_migrate(self):
        self._mock_action()
        self.management.migrate(1)
        self.assertEqual(1, self.management._action.call_count)
        self.assertEqual({'migrate': {}}, self.body_)

    def test_migrate_to_host(self):
        hostname = 'hostname2'
        self._mock_action()
        self.management.migrate(1, host=hostname)
        self.assertEqual(1, self.management._action.call_count)
        self.assertEqual({'migrate': {'host': hostname}}, self.body_)

    def test_update(self):
        self._mock_action()
        self.management.update(1)
        self.assertEqual(1, self.management._action.call_count)
        self.assertEqual({'update': {}}, self.body_)

    def test_reset_task_status(self):
        self._mock_action()
        self.management.reset_task_status(1)
        self.assertEqual(1, self.management._action.call_count)
        self.assertEqual({'reset-task-status': {}}, self.body_)