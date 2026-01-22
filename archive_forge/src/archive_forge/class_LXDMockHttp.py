import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_LXD
from libcloud.container.base import Container, ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.lxd import (
class LXDMockHttp(MockHttp):
    fixtures = ContainerFileFixtures('lxd')
    version = None

    def _version(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('linux_124/version.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def root(self, method, url, body, headers):
        json = self.fixtures.load('linux_124/endpoints_sucess.json')
        return (httplib.OK, json, {}, httplib.responses[httplib.OK])

    def _linux_124(self, method, url, body, headers):
        return (httplib.OK, self.fixtures.load('linux_124/version.json'), {}, httplib.responses[httplib.OK])

    def _linux_124_images(self, method, url, body, headers):
        return (httplib.OK, self.fixtures.load('linux_124/images.json'), {}, httplib.responses[httplib.OK])

    def _linux_124_images_54c8caac1f61901ed86c68f24af5f5d3672bdc62c71d04f06df3a59e95684473(self, method, url, body, headers):
        return (httplib.OK, self.fixtures.load('linux_124/image.json'), {}, httplib.responses[httplib.OK])

    def _linux_124_containers(self, method, url, body, headers):
        if method == 'GET':
            return (httplib.OK, self.fixtures.load('linux_124/containers.json'), {}, httplib.responses[httplib.OK])
        elif method == 'POST' or method == 'PUT':
            return (httplib.OK, self.fixtures.load('linux_124/background_op.json'), {}, httplib.responses[httplib.OK])

    def _linux_124_containers_first_lxd_container(self, method, url, body, headers):
        return (httplib.OK, self.fixtures.load('linux_124/first_lxd_container.json'), {}, httplib.responses[httplib.OK])

    def _linux_124_containers_second_lxd_container(self, method, url, body, headers):
        if method == 'PUT' or method == 'DELETE':
            json = self.fixtures.load('linux_124/background_op.json')
            return (httplib.OK, json, {}, httplib.responses[httplib.OK])
        elif method == 'GET':
            return (httplib.OK, self.fixtures.load('linux_124/second_lxd_container.json'), {}, httplib.responses[httplib.OK])

    def _linux_124_containers_first_lxd_container_state(self, method, url, body, headers):
        if method == 'PUT' or method == 'DELETE':
            json = self.fixtures.load('linux_124/background_op.json')
            return (httplib.OK, json, {}, httplib.responses[httplib.OK])
        elif method == 'GET':
            json = self.fixtures.load('linux_124/first_lxd_container.json')
            return (httplib.OK, json, {}, httplib.responses[httplib.OK])

    def _linux_124_containers_second_lxd_container_state(self, method, url, body, headers):
        if method == 'PUT' or method == 'DELETE':
            json = self.fixtures.load('linux_124/background_op.json')
            return (httplib.OK, json, {}, httplib.responses[httplib.OK])
        elif method == 'GET':
            json = self.fixtures.load('linux_124/second_lxd_container.json')
            return (httplib.OK, json, {}, httplib.responses[httplib.OK])

    def _linux_124_operations_1_wait(self, method, url, body, header):
        return (httplib.OK, self.fixtures.load('linux_124/operation_1_wait.json'), {}, httplib.responses[httplib.OK])

    def _linux_124_storage_pools(self, method, url, body, header):
        if method == 'GET':
            json = self.fixtures.load('linux_124/storage_pools.json')
            return (httplib.OK, json, {}, httplib.responses[httplib.OK])

    def _linux_124_storage_pools_pool1(self, method, url, body, header):
        if method == 'GET':
            json = self.fixtures.load('linux_124/storage_pool_1.json')
            return (httplib.OK, json, {}, httplib.responses[httplib.OK])
        elif method == 'DELETE':
            return (httplib.OK, self.fixtures.load('linux_124/storage_pool_delete_sucess.json'), {}, httplib.responses[httplib.OK])

    def _linux_124_storage_pools_pool2(self, method, url, body, header):
        if method == 'GET':
            json = self.fixtures.load('linux_124/storage_pool_2.json')
            return (httplib.OK, json, {}, httplib.responses[httplib.OK])
        elif method == 'DELETE':
            return (httplib.OK, self.fixtures.load('linux_124/storage_pool_delete_fail.json'), {}, httplib.responses[httplib.OK])

    def _linux_124_storage_pools_pool3(self, method, url, body, header):
        if method == 'GET':
            json = self.fixtures.load('linux_124/no_meta_pool.json')
            return (httplib.OK, json, {}, httplib.responses[httplib.OK])
    "\n    def _vlinux_124_containers_create(\n        self, method, url, body, headers):\n        return (httplib.OK, self.fixtures.load('linux_124/create_container.json'), {}, httplib.responses[httplib.OK])\n\n    def _vlinux_124_containers_a68c1872c74630522c7aa74b85558b06824c5e672cee334296c50fb209825303(\n        self, method, url, body, headers):\n        return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.OK])\n\n    def _vlinux_124_containers_a68c1872c74630522c7aa74b85558b06824c5e672cee334296c50fb209825303_start(\n        self, method, url, body, headers):\n        return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.OK])\n\n    def _vlinux_124_containers_a68c1872c74630522c7aa74b85558b06824c5e672cee334296c50fb209825303_restart(\n        self, method, url, body, headers):\n        return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.OK])\n\n    def _vlinux_124_containers_a68c1872c74630522c7aa74b85558b06824c5e672cee334296c50fb209825303_rename(\n        self, method, url, body, headers):\n        return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.OK])\n\n    def _vlinux_124_containers_a68c1872c74630522c7aa74b85558b06824c5e672cee334296c50fb209825303_stop(\n        self, method, url, body, headers):\n        return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.OK])\n\n    def _vlinux_124_containers_a68c1872c74630522c7aa74b85558b06824c5e672cee334296c50fb209825303_json(\n        self, method, url, body, headers):\n        return (httplib.OK, self.fixtures.load('linux_124/container_a68.json'), {}, httplib.responses[httplib.OK])\n\n    def _vlinux_124_containers_a68c1872c74630522c7aa74b85558b06824c5e672cee334296c50fb209825303_logs(\n        self, method, url, body, headers):\n        return (httplib.OK, self.fixtures.load('linux_124/logs.txt'), {'content-type': 'text/plain'},\n                httplib.responses[httplib.OK])\n    "