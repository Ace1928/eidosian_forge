import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_LXD
from libcloud.container.base import Container, ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.lxd import (
class LXDContainerDriverTestCase(unittest.TestCase):
    """
    Unit tests for LXDContainerDriver
    """

    def setUp(self):
        versions = ('linux_124',)
        self.drivers = []
        for version in versions:
            LXDContainerDriver.connectionCls.conn_class = LXDMockHttp
            LXDMockHttp.type = None
            LXDMockHttp.use_param = 'a'
            driver = LXDContainerDriver(*CONTAINER_PARAMS_LXD)
            driver.connectionCls.conn_class = LXDMockHttp
            driver.version = version
            self.drivers.append(driver)

    def test_ex_get_api_endpoints_trusted(self):
        for driver in self.drivers:
            api = driver.ex_get_api_endpoints()
            self.assertEqual(api[0], driver.version)

    def test_ex_get_server_configuration(self):
        for driver in self.drivers:
            server_config = driver.ex_get_server_configuration()
            self.assertIsInstance(server_config, LXDServerInfo)
            self.assertEqual(server_config.api_extensions, [])
            self.assertEqual(server_config.api_status, 'stable')
            self.assertEqual(server_config.api_version, 'linux_124')
            self.assertEqual(server_config.auth, 'guest')
            self.assertEqual(server_config.public, False)

    def test_list_images(self):
        img_id = '54c8caac1f61901ed86c68f24af5f5d3672bdc62c71d04f06df3a59e95684473'
        for driver in self.drivers:
            images = driver.list_images()
            self.assertEqual(len(images), 1)
            self.assertIsInstance(images[0], ContainerImage)
            self.assertEqual(images[0].id, img_id)
            self.assertEqual(images[0].name, 'trusty')

    def test_list_containers(self):
        for driver in self.drivers:
            containers = driver.list_containers()
            self.assertEqual(len(containers), 2)
            self.assertIsInstance(containers[0], Container)
            self.assertIsInstance(containers[1], Container)
            self.assertEqual(containers[0].name, 'first_lxd_container')
            self.assertEqual(containers[1].name, 'second_lxd_container')

    def test_get_container(self):
        for driver in self.drivers:
            container = driver.get_container(id='second_lxd_container')
            self.assertIsInstance(container, Container)
            self.assertEqual(container.name, 'second_lxd_container')
            self.assertEqual(container.id, 'second_lxd_container')
            self.assertEqual(container.state, 'stopped')

    def test_start_container(self):
        for driver in self.drivers:
            container = driver.get_container(id='first_lxd_container')
            container.start()
            self.assertEqual(container.state, 'running')

    def test_stop_container(self):
        for driver in self.drivers:
            container = driver.get_container(id='second_lxd_container')
            container.stop()
            self.assertEqual(container.state, 'stopped')

    def test_restart_container(self):
        for driver in self.drivers:
            container = driver.get_container(id='second_lxd_container')
            container.restart()

    def test_delete_container(self):
        for driver in self.drivers:
            container = driver.get_container(id='second_lxd_container')
            container.destroy()

    def test_deploy_container(self):
        for driver in self.drivers:
            image = ContainerImage(id=None, name=None, path=None, version=None, driver=driver)
            container = driver.deploy_container(name='first_lxd_container', image=image, parameters='{"source":{"type":"image", "fingerprint":"7ed08b435c92cd8a8a884c88e8722f2e7546a51e891982a90ea9c15619d7df9b"}}')
            self.assertIsInstance(container, Container)
            self.assertEqual(container.name, 'first_lxd_container')

    def test_install_image_no_dict(self):
        with self.assertRaises(LXDAPIException) as exc:
            for driver in self.drivers:
                driver.install_image(path=None)
                self.assertEqual(str(exc), 'Install an image for LXD requires specification of image_data')

    def test_list_storage_pools(self):
        for driver in self.drivers:
            pools = driver.ex_list_storage_pools()
            self.assertEqual(len(pools), 2)
            self.assertIsInstance(pools[0], LXDStoragePool)
            self.assertIsInstance(pools[1], LXDStoragePool)
            self.assertEqual(pools[0].name, 'pool1')
            self.assertEqual(pools[1].name, 'pool2')

    def test_get_storage_pool_no_metadata(self):
        with self.assertRaises(LXDAPIException) as exc:
            for driver in self.drivers:
                driver.ex_get_storage_pool(id='pool3')
                self.assertEqual(str(exc), 'Storage pool with name pool3 has no data')

    def test_delete_storage_pool(self):
        for driver in self.drivers:
            driver.ex_delete_storage_pool(id='pool1')

    def test_delete_storage_pool_fail(self):
        with self.assertRaises(LXDAPIException):
            for driver in self.drivers:
                driver.ex_delete_storage_pool(id='pool2')