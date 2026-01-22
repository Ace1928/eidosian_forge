import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
@ddt.ddt
class ConnectionPropertiesDecoratorsTestCase(base.TestCase):

    def test__symlink_name_from_device_path(self):
        """Get symlink for non replicated device."""
        dev_name = '/dev/nvme0n1'
        res = utils._symlink_name_from_device_path(dev_name)
        self.assertEqual('/dev/disk/by-id/os-brick+dev+nvme0n1', res)

    def test__symlink_name_from_device_path_raid(self):
        """Get symlink for replicated device."""
        dev_name = '/dev/md/alias'
        res = utils._symlink_name_from_device_path(dev_name)
        self.assertEqual('/dev/disk/by-id/os-brick+dev+md+alias', res)

    def test__device_path_from_symlink(self):
        """Get device name for non replicated symlink."""
        symlink = '/dev/disk/by-id/os-brick+dev+nvme0n1'
        res = utils._device_path_from_symlink(symlink)
        self.assertEqual('/dev/nvme0n1', res)

    def test__device_path_from_symlink_raid(self):
        """Get device name for replicated symlink."""
        symlink = '/dev/disk/by-id/os-brick+dev+md+alias'
        res = utils._device_path_from_symlink(symlink)
        self.assertEqual('/dev/md/alias', res)

    def test__device_path_from_symlink_file_handle(self):
        """Get device name for a file handle (eg: RBD)."""
        handle = io.StringIO()
        res = utils._device_path_from_symlink(handle)
        self.assertEqual(handle, res)

    @ddt.data(({}, {'type': 'block', 'path': '/dev/sda'}), ({'encrypted': False}, {'type': 'block', 'path': '/dev/sda'}), ({'encrypted': False}, {'type': 'block', 'path': b'/dev/sda'}), ({'encrypted': True}, {'type': 'block', 'path': io.StringIO()}))
    @ddt.unpack
    @mock.patch('os_brick.utils._symlink_name_from_device_path')
    @mock.patch('os.path.realpath')
    @mock.patch('os_brick.privileged.rootwrap.link_root')
    def test_connect_volume_prepare_result_non_encrypted(self, conn_props, result, mock_link, mock_path, mock_get_symlink):
        """Test decorator for non encrypted devices or non host devices."""
        testing_self = mock.Mock()
        testing_self.connect_volume.return_value = result
        func = utils.connect_volume_prepare_result(testing_self.connect_volume)
        res = func(testing_self, conn_props)
        self.assertEqual(testing_self.connect_volume.return_value, res)
        testing_self.connect_volume.assert_called_once_with(testing_self, conn_props)
        mock_path.assert_not_called()
        mock_get_symlink.assert_not_called()
        mock_link.assert_not_called()

    @ddt.data('/dev/md/alias', b'/dev/md/alias')
    @mock.patch('os_brick.utils._symlink_name_from_device_path')
    @mock.patch('os.path.realpath')
    @mock.patch('os_brick.privileged.rootwrap.link_root')
    def test_connect_volume_prepare_result_encrypted(self, connector_path, mock_link, mock_path, mock_get_symlink):
        """Test decorator for encrypted device."""
        real_device = '/dev/md-6'
        expected_symlink = '/dev/disk/by-id/os-brick_dev_md_alias'
        mock_path.return_value = real_device
        mock_get_symlink.return_value = expected_symlink
        testing_self = mock.Mock()
        testing_self.connect_volume.return_value = {'type': 'block', 'path': connector_path}
        conn_props = {'encrypted': True}
        func = utils.connect_volume_prepare_result(testing_self.connect_volume)
        res = func(testing_self, conn_props)
        self.assertEqual({'type': 'block', 'path': expected_symlink}, res)
        testing_self.connect_volume.assert_called_once_with(testing_self, conn_props)
        expected_connector_path = utils.convert_str(connector_path)
        mock_get_symlink.assert_called_once_with(expected_connector_path)
        mock_link.assert_called_once_with(real_device, expected_symlink, force=True)

    @ddt.data({}, {'encrypted': False}, {'encrypted': True})
    @mock.patch('os_brick.utils._symlink_name_from_device_path')
    @mock.patch('os.path.realpath')
    @mock.patch('os_brick.privileged.rootwrap.link_root')
    def test_connect_volume_prepare_result_connect_fail(self, conn_props, mock_link, mock_path, mock_get_symlink):
        """Test decorator when decorated function fails."""
        testing_self = mock.Mock()
        testing_self.connect_volume.side_effect = ValueError
        func = utils.connect_volume_prepare_result(testing_self.connect_volume)
        self.assertRaises(ValueError, func, testing_self, conn_props)
        mock_link.assert_not_called()
        mock_path.assert_not_called()
        mock_get_symlink.assert_not_called()

    @mock.patch('os_brick.utils._symlink_name_from_device_path')
    @mock.patch('os.path.realpath')
    @mock.patch('os_brick.privileged.rootwrap.link_root')
    def test_connect_volume_prepare_result_symlink_fail(self, mock_link, mock_path, mock_get_symlink):
        """Test decorator for encrypted device failing on the symlink."""
        real_device = '/dev/md-6'
        connector_path = '/dev/md/alias'
        expected_symlink = '/dev/disk/by-id/os-brick_dev_md_alias'
        mock_path.return_value = real_device
        mock_get_symlink.return_value = expected_symlink
        testing_self = mock.Mock()
        connect_result = {'type': 'block', 'path': connector_path}
        mock_link.side_effect = ValueError
        testing_self.connect_volume.return_value = connect_result
        conn_props = {'encrypted': True}
        func = utils.connect_volume_prepare_result(testing_self.connect_volume)
        self.assertRaises(ValueError, func, testing_self, conn_props)
        testing_self.connect_volume.assert_called_once_with(testing_self, conn_props)
        mock_get_symlink.assert_called_once_with(connector_path)
        mock_link.assert_called_once_with(real_device, expected_symlink, force=True)
        testing_self.disconnect_volume.assert_called_once_with(connect_result, force=True, ignore_errors=True)

    @ddt.data(({'device_path': '/dev/md/alias'}, {}), ({'device_path': '/dev/md/alias', 'encrypted': False}, None), ({'device_path': '/dev/md/alias'}, {'path': '/dev/md/alias'}), ({'device_path': '/dev/md/alias', 'encrypted': False}, {'path': '/dev/md/alias'}), ({'device_path': io.StringIO(), 'encrypted': True}, None), ({'device_path': '/dev/disk/by-id/wwn-...', 'encrypted': True}, None))
    @ddt.unpack
    @mock.patch('os_brick.utils._device_path_from_symlink')
    @mock.patch('os_brick.privileged.rootwrap.unlink_root')
    def test_connect_volume_undo_prepare_result_non_custom_link(outer_self, conn_props, dev_info, mock_unlink, mock_dev_path):

        class Test(object):

            @utils.connect_volume_undo_prepare_result(unlink_after=True)
            def disconnect_volume(self, connection_properties, device_info, force=False, ignore_errors=False):
                outer_self.assertEqual(conn_props, connection_properties)
                outer_self.assertEqual(dev_info, device_info)
                return 'disconnect_volume'

            @utils.connect_volume_undo_prepare_result
            def extend_volume(self, connection_properties):
                outer_self.assertEqual(conn_props, connection_properties)
                return 'extend_volume'
        path = conn_props['device_path']
        mock_dev_path.return_value = path
        t = Test()
        res = t.disconnect_volume(conn_props, dev_info)
        outer_self.assertEqual('disconnect_volume', res)
        res = t.extend_volume(conn_props)
        outer_self.assertEqual('extend_volume', res)
        if conn_props.get('encrypted'):
            outer_self.assertEqual(2, mock_dev_path.call_count)
            mock_dev_path.assert_has_calls((mock.call(path), mock.call(path)))
        else:
            mock_dev_path.assert_not_called()
        mock_unlink.assert_not_called()

    @mock.patch('os_brick.utils._device_path_from_symlink')
    @mock.patch('os_brick.privileged.rootwrap.unlink_root')
    def test_connect_volume_undo_prepare_result_encrypted_disconnect(outer_self, mock_unlink, mock_dev_path):
        connector_path = '/dev/md/alias'
        mock_dev_path.return_value = connector_path
        symlink_path = '/dev/disk/by-id/os-brick_dev_md_alias'
        mock_unlink.side_effect = ValueError

        class Test(object):

            @utils.connect_volume_undo_prepare_result(unlink_after=True)
            def disconnect_volume(self, connection_properties, device_info, force=False, ignore_errors=False):
                outer_self.assertEqual(connector_path, connection_properties['device_path'])
                outer_self.assertEqual(connector_path, device_info['path'])
                return 'disconnect_volume'
        conn_props = {'target_portal': '198.72.124.185:3260', 'target_iqn': 'iqn.2010-10.org.openstack:volume-uuid', 'target_lun': 0, 'encrypted': True, 'device_path': symlink_path}
        dev_info = {'type': 'block', 'path': symlink_path}
        t = Test()
        res = t.disconnect_volume(conn_props, dev_info)
        outer_self.assertEqual('disconnect_volume', res)
        mock_dev_path.assert_called_once_with(symlink_path)
        mock_unlink.assert_called_once_with(symlink_path)

    @mock.patch('os_brick.utils._device_path_from_symlink')
    @mock.patch('os_brick.privileged.rootwrap.unlink_root')
    def test_connect_volume_undo_prepare_result_encrypted_extend(outer_self, mock_unlink, mock_dev_path):
        connector_path = '/dev/md/alias'
        mock_dev_path.return_value = connector_path
        symlink_path = '/dev/disk/by-id/os-brick_dev_md_alias'
        mock_unlink.side_effect = ValueError

        class Test(object):

            @utils.connect_volume_undo_prepare_result
            def extend_volume(self, connection_properties):
                outer_self.assertEqual(connector_path, connection_properties['device_path'])
                return 'extend_volume'
        conn_props = {'target_portal': '198.72.124.185:3260', 'target_iqn': 'iqn.2010-10.org.openstack:volume-uuid', 'target_lun': 0, 'encrypted': True, 'device_path': symlink_path}
        t = Test()
        res = t.extend_volume(conn_props)
        outer_self.assertEqual('extend_volume', res)
        mock_dev_path.assert_called_once_with(symlink_path)
        mock_unlink.assert_not_called()