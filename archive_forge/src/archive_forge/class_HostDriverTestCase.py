import errno
from unittest import mock
from os_brick.initiator import host_driver
from os_brick.tests import base
class HostDriverTestCase(base.TestCase):

    def test_get_all_block_devices(self):
        fake_dev = ['device1', 'device2']
        expected = ['/dev/disk/by-path/' + dev for dev in fake_dev]
        driver = host_driver.HostDriver()
        with mock.patch('os.listdir', return_value=fake_dev):
            actual = driver.get_all_block_devices()
        self.assertEqual(expected, actual)

    def test_get_all_block_devices_when_oserror_is_enoent(self):
        driver = host_driver.HostDriver()
        oserror = OSError(errno.ENOENT, '')
        with mock.patch('os.listdir', side_effect=oserror):
            block_devices = driver.get_all_block_devices()
        self.assertEqual([], block_devices)

    def test_get_all_block_devices_when_oserror_is_not_enoent(self):
        driver = host_driver.HostDriver()
        oserror = OSError(errno.ENOMEM, '')
        with mock.patch('os.listdir', side_effect=oserror):
            self.assertRaises(OSError, driver.get_all_block_devices)