import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_RANCHER
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.rancher import RancherContainerDriver
def _v1_containers_1i31(self, method, url, body, headers):
    if method == 'GET':
        return (httplib.OK, self.fixtures.load('deploy_container.json'), {}, httplib.responses[httplib.OK])
    elif method == 'DELETE' or '?action=stop' in url:
        return (httplib.OK, self.fixtures.load('stop_container.json'), {}, httplib.responses[httplib.OK])
    elif '?action=start' in url:
        return (httplib.OK, self.fixtures.load('start_container.json'), {}, httplib.responses[httplib.OK])
    else:
        return (httplib.OK, self.fixtures.load('deploy_container.json'), {}, httplib.responses[httplib.OK])