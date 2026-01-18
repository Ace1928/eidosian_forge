import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_LXD
from libcloud.container.base import Container, ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.lxd import (

    def _vlinux_124_containers_create(
        self, method, url, body, headers):
        return (httplib.OK, self.fixtures.load('linux_124/create_container.json'), {}, httplib.responses[httplib.OK])

    def _vlinux_124_containers_a68c1872c74630522c7aa74b85558b06824c5e672cee334296c50fb209825303(
        self, method, url, body, headers):
        return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.OK])

    def _vlinux_124_containers_a68c1872c74630522c7aa74b85558b06824c5e672cee334296c50fb209825303_start(
        self, method, url, body, headers):
        return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.OK])

    def _vlinux_124_containers_a68c1872c74630522c7aa74b85558b06824c5e672cee334296c50fb209825303_restart(
        self, method, url, body, headers):
        return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.OK])

    def _vlinux_124_containers_a68c1872c74630522c7aa74b85558b06824c5e672cee334296c50fb209825303_rename(
        self, method, url, body, headers):
        return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.OK])

    def _vlinux_124_containers_a68c1872c74630522c7aa74b85558b06824c5e672cee334296c50fb209825303_stop(
        self, method, url, body, headers):
        return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.OK])

    def _vlinux_124_containers_a68c1872c74630522c7aa74b85558b06824c5e672cee334296c50fb209825303_json(
        self, method, url, body, headers):
        return (httplib.OK, self.fixtures.load('linux_124/container_a68.json'), {}, httplib.responses[httplib.OK])

    def _vlinux_124_containers_a68c1872c74630522c7aa74b85558b06824c5e672cee334296c50fb209825303_logs(
        self, method, url, body, headers):
        return (httplib.OK, self.fixtures.load('linux_124/logs.txt'), {'content-type': 'text/plain'},
                httplib.responses[httplib.OK])
    