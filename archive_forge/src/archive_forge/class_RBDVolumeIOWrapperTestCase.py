from unittest import mock
from os_brick import exception
from os_brick.initiator import linuxrbd
from os_brick.tests import base
from os_brick import utils
class RBDVolumeIOWrapperTestCase(base.TestCase):

    def setUp(self):
        super(RBDVolumeIOWrapperTestCase, self).setUp()
        self.mock_volume = mock.Mock()
        self.mock_volume_wrapper = linuxrbd.RBDVolumeIOWrapper(self.mock_volume)
        self.data_length = 1024
        self.full_data = 'abcd' * 256
        self._rbd_lib = self.patch('os_brick.initiator.linuxrbd.rbd')
        self._rbd_lib.InvalidArgument = InvalidArgument

    def test_init(self):
        self.assertEqual(self.mock_volume, self.mock_volume_wrapper._rbd_volume)
        self.assertEqual(0, self.mock_volume_wrapper._offset)

    def test_inc_offset(self):
        self.mock_volume_wrapper._inc_offset(10)
        self.mock_volume_wrapper._inc_offset(10)
        self.assertEqual(20, self.mock_volume_wrapper._offset)

    def test_read(self):

        def mock_read(offset, length):
            return self.full_data[offset:length]
        self.mock_volume.image.read.side_effect = mock_read
        self.mock_volume.image.size.return_value = self.data_length
        data = self.mock_volume_wrapper.read()
        self.assertEqual(self.full_data, data)
        data = self.mock_volume_wrapper.read()
        self.assertEqual(b'', data)
        self.mock_volume_wrapper.seek(0)
        data = self.mock_volume_wrapper.read()
        self.assertEqual(self.full_data, data)
        self.mock_volume_wrapper.seek(0)
        data = self.mock_volume_wrapper.read(10)
        self.assertEqual(self.full_data[:10], data)

    def test_write(self):
        self.mock_volume_wrapper.write(self.full_data)
        self.assertEqual(1024, self.mock_volume_wrapper._offset)

    def test_seekable(self):
        self.assertTrue(self.mock_volume_wrapper.seekable)

    def test_seek(self):
        self.assertEqual(0, self.mock_volume_wrapper._offset)
        self.mock_volume_wrapper.seek(10)
        self.assertEqual(10, self.mock_volume_wrapper._offset)
        self.mock_volume_wrapper.seek(10)
        self.assertEqual(10, self.mock_volume_wrapper._offset)
        self.mock_volume_wrapper.seek(10, 1)
        self.assertEqual(20, self.mock_volume_wrapper._offset)
        self.mock_volume_wrapper.seek(0)
        self.mock_volume_wrapper.write(self.full_data)
        self.mock_volume.image.size.return_value = self.data_length
        self.mock_volume_wrapper.seek(0)
        self.assertEqual(0, self.mock_volume_wrapper._offset)
        self.mock_volume_wrapper.seek(10, 2)
        self.assertEqual(self.data_length + 10, self.mock_volume_wrapper._offset)
        self.mock_volume_wrapper.seek(-10, 2)
        self.assertEqual(self.data_length - 10, self.mock_volume_wrapper._offset)
        self.assertRaises(IOError, self.mock_volume_wrapper.seek, 0, 3)
        self.assertRaises(IOError, self.mock_volume_wrapper.seek, -1)
        self.assertEqual(self.data_length - 10, self.mock_volume_wrapper._offset)

    def test_tell(self):
        self.assertEqual(0, self.mock_volume_wrapper.tell())
        self.mock_volume_wrapper._inc_offset(10)
        self.assertEqual(10, self.mock_volume_wrapper.tell())

    def test_flush(self):
        with mock.patch.object(linuxrbd, 'LOG') as mock_logger:
            self.mock_volume.image.flush = mock.Mock()
            self.mock_volume_wrapper.flush()
            self.assertEqual(1, self.mock_volume.image.flush.call_count)
            self.mock_volume.image.require_not_closed.assert_called_once_with()
            self.mock_volume.image.flush.reset_mock()
            self.mock_volume.image.require_not_closed.reset_mock()
            self.mock_volume.image.flush.side_effect = AttributeError
            self.mock_volume_wrapper.flush()
            self.assertEqual(1, self.mock_volume.image.flush.call_count)
            self.assertEqual(1, mock_logger.warning.call_count)
            self.mock_volume.image.require_not_closed.assert_called_once_with()

    def test_flush_closed_image(self):
        """Test when image is closed but wrapper isn't"""
        with mock.patch.object(linuxrbd, 'LOG') as mock_logger:
            self.mock_volume.image.require_not_closed.side_effect = InvalidArgument
            self.mock_volume.image.flush = mock.Mock()
            self.mock_volume_wrapper.flush()
            self.mock_volume.image.flush.assert_not_called()
            self.assertEqual(1, mock_logger.warning.call_count)
            log_msg = mock_logger.warning.call_args[0][0]
            self.assertTrue(log_msg.startswith("RBDVolumeIOWrapper's underlying image"))
            self.mock_volume.image.require_not_closed.assert_called_once_with()

    def test_flush_on_closed(self):
        self.mock_volume_wrapper.close()
        self.mock_volume.image.flush.assert_called_once_with()
        self.assertTrue(self.mock_volume_wrapper.closed)
        self.mock_volume.image.flush.reset_mock()
        self.assertRaises(ValueError, self.mock_volume_wrapper.flush)
        self.mock_volume.image.flush.assert_not_called()
        self.mock_volume.image.require_not_closed.assert_called_once_with()

    def test_flush_on_image_closed(self):
        self.mock_volume.image.require_not_closed.side_effect = InvalidArgument
        self.mock_volume_wrapper.close()
        self.mock_volume.image.flush.assert_not_called()
        self.assertTrue(self.mock_volume_wrapper.closed)
        self.mock_volume.image.close.assert_called_once_with()
        self.mock_volume.image.require_not_closed.assert_called_once_with()

    def test_fileno(self):
        self.assertRaises(IOError, self.mock_volume_wrapper.fileno)

    @mock.patch('os_brick.initiator.linuxrbd.rbd')
    @mock.patch('os_brick.initiator.linuxrbd.rados')
    @mock.patch.object(linuxrbd.RBDClient, 'disconnect')
    def test_close(self, rbd_disconnect, mock_rados, mock_rbd):
        rbd_client = linuxrbd.RBDClient('user', 'pool')
        rbd_volume = linuxrbd.RBDVolume(rbd_client, 'volume')
        rbd_handle = linuxrbd.RBDVolumeIOWrapper(linuxrbd.RBDImageMetadata(rbd_volume, 'pool', 'user', None))
        with mock.patch.object(rbd_volume, 'closed', False):
            rbd_handle.close()
        self.assertEqual(1, rbd_disconnect.call_count)
        self.assertTrue(rbd_handle.closed)
        rbd_handle.close()
        self.assertEqual(1, rbd_disconnect.call_count)