import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_KUBERNETES
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
from libcloud.container.drivers.kubernetes import (
def _apis_apps_v1_deployments(self, method, url, body, headers):
    body = self.fixtures.load('_apis_apps_v1_deployments.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])