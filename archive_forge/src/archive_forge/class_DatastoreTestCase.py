from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
class DatastoreTestCase(base.TestCase):
    """Test the Datastore object."""

    def test_ds(self):
        ds = datastore.Datastore('fake_ref', 'ds_name', 2 * units.Gi, 1 * units.Gi, 1 * units.Gi)
        self.assertEqual('ds_name', ds.name)
        self.assertEqual('fake_ref', ds.ref)
        self.assertEqual(2 * units.Gi, ds.capacity)
        self.assertEqual(1 * units.Gi, ds.freespace)
        self.assertEqual(1 * units.Gi, ds.uncommitted)

    def test_ds_invalid_space(self):
        self.assertRaises(ValueError, datastore.Datastore, 'fake_ref', 'ds_name', 1 * units.Gi, 2 * units.Gi)
        self.assertRaises(ValueError, datastore.Datastore, 'fake_ref', 'ds_name', None, 2 * units.Gi)

    def test_ds_no_capacity_no_freespace(self):
        ds = datastore.Datastore('fake_ref', 'ds_name')
        self.assertIsNone(ds.capacity)
        self.assertIsNone(ds.freespace)

    def test_ds_invalid(self):
        self.assertRaises(ValueError, datastore.Datastore, None, 'ds_name')
        self.assertRaises(ValueError, datastore.Datastore, 'fake_ref', None)

    def test_build_path(self):
        ds = datastore.Datastore('fake_ref', 'ds_name')
        ds_path = ds.build_path('some_dir', 'foo.vmdk')
        self.assertEqual('[ds_name] some_dir/foo.vmdk', str(ds_path))

    def test_build_url(self):
        ds = datastore.Datastore('fake_ref', 'ds_name')
        path = 'images/ubuntu.vmdk'
        self.assertRaises(ValueError, ds.build_url, 'https', '10.0.0.2', path)
        ds.datacenter = mock.Mock()
        ds.datacenter.name = 'dc_path'
        ds_url = ds.build_url('https', '10.0.0.2', path)
        self.assertEqual(ds_url.datastore_name, 'ds_name')
        self.assertEqual(ds_url.datacenter_path, 'dc_path')
        self.assertEqual(ds_url.path, path)

    def test_get_summary(self):
        ds_ref = vim_util.get_moref('ds-0', 'Datastore')
        ds = datastore.Datastore(ds_ref, 'ds-name')
        summary = mock.sentinel.summary
        session = mock.Mock()
        session.invoke_api = mock.Mock()
        session.invoke_api.return_value = summary
        ret = ds.get_summary(session)
        self.assertEqual(summary, ret)
        session.invoke_api.assert_called_once_with(vim_util, 'get_object_property', session.vim, ds.ref, 'summary')

    def _test_get_connected_hosts(self, in_maintenance_mode, m1_accessible=True):
        session = mock.Mock()
        ds_ref = vim_util.get_moref('ds-0', 'Datastore')
        ds = datastore.Datastore(ds_ref, 'ds-name')
        ds.get_summary = mock.Mock()
        ds.get_summary.return_value.accessible = False
        self.assertEqual([], ds.get_connected_hosts(session))
        ds.get_summary.return_value.accessible = True
        m1 = HostMount('m1', MountInfo('readWrite', True, m1_accessible))
        m2 = HostMount('m2', MountInfo('read', True, True))
        m3 = HostMount('m3', MountInfo('readWrite', False, True))
        m4 = HostMount('m4', MountInfo('readWrite', True, False))
        ds.get_summary.assert_called_once_with(session)

        class Prop(object):
            DatastoreHostMount = [m1, m2, m3, m4]

        class HostRuntime(object):
            inMaintenanceMode = in_maintenance_mode

        class HostProp(object):
            name = 'runtime'
            val = HostRuntime()

        class Object(object):
            obj = 'm1'
            propSet = [HostProp()]

        class Runtime(object):
            objects = [Object()]
        session.invoke_api = mock.Mock(side_effect=[Prop(), Runtime()])
        hosts = ds.get_connected_hosts(session)
        calls = [mock.call(vim_util, 'get_object_property', session.vim, ds_ref, 'host')]
        if m1_accessible:
            calls.append(mock.call(vim_util, 'get_properties_for_a_collection_of_objects', session.vim, 'HostSystem', ['m1'], ['runtime']))
        self.assertEqual(calls, session.invoke_api.mock_calls)
        return hosts

    def test_get_connected_hosts(self):
        hosts = self._test_get_connected_hosts(False)
        self.assertEqual(1, len(hosts))
        self.assertEqual('m1', hosts.pop())

    def test_get_connected_hosts_in_maintenance(self):
        hosts = self._test_get_connected_hosts(True)
        self.assertEqual(0, len(hosts))

    def test_get_connected_hosts_ho_hosts(self):
        hosts = self._test_get_connected_hosts(False, False)
        self.assertEqual(0, len(hosts))

    def test_is_datastore_mount_usable(self):
        m = MountInfo('readWrite', True, True)
        self.assertTrue(datastore.Datastore.is_datastore_mount_usable(m))
        m = MountInfo('read', True, True)
        self.assertFalse(datastore.Datastore.is_datastore_mount_usable(m))
        m = MountInfo('readWrite', False, True)
        self.assertFalse(datastore.Datastore.is_datastore_mount_usable(m))
        m = MountInfo('readWrite', True, False)
        self.assertFalse(datastore.Datastore.is_datastore_mount_usable(m))
        m = MountInfo('readWrite', False, False)
        self.assertFalse(datastore.Datastore.is_datastore_mount_usable(m))
        m = MountInfo('readWrite', None, None)
        self.assertFalse(datastore.Datastore.is_datastore_mount_usable(m))
        m = MountInfo('readWrite', None, True)
        self.assertFalse(datastore.Datastore.is_datastore_mount_usable(m))