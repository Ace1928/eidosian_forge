import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_DOCKER
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.docker import DockerContainerDriver
def _vmac_124_containers_a68c1872c74630522c7aa74b85558b06824c5e672cee334296c50fb209825303_json(self, method, url, body, headers):
    return (httplib.OK, self.fixtures.load('linux_124/container_a68.json'), {}, httplib.responses[httplib.OK])