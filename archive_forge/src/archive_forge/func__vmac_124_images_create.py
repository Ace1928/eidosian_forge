import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_DOCKER
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.docker import DockerContainerDriver
def _vmac_124_images_create(self, method, url, body, headers):
    return (httplib.OK, self.fixtures.load('mac_124/create_image.txt'), {'Content-Type': 'application/json', 'transfer-encoding': 'chunked'}, httplib.responses[httplib.OK])