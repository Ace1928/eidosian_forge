import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_RANCHER
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.rancher import RancherContainerDriver
class RancherMockHttp(MockHttp):
    fixtures = ContainerFileFixtures('rancher')

    def _v1_environments(self, method, url, body, headers):
        if method == 'GET':
            return (httplib.OK, self.fixtures.load('ex_list_stacks.json'), {}, httplib.responses[httplib.OK])
        else:
            return (httplib.OK, self.fixtures.load('ex_deploy_stack.json'), {}, httplib.responses[httplib.OK])

    def _v1_environments_1e9(self, method, url, body, headers):
        return (httplib.OK, self.fixtures.load('ex_deploy_stack.json'), {}, httplib.responses[httplib.OK])

    def _v1_environments_1e10(self, method, url, body, headers):
        return (httplib.OK, self.fixtures.load('ex_destroy_stack.json'), {}, httplib.responses[httplib.OK])

    def _v1_environments_1e1(self, method, url, body, headers):
        return (httplib.OK, self.fixtures.load('ex_activate_stack.json'), {}, httplib.responses[httplib.OK])

    def _v1_services(self, method, url, body, headers):
        if '?healthState=healthy' in url:
            return (httplib.OK, self.fixtures.load('ex_search_services.json'), {}, httplib.responses[httplib.OK])
        elif method == 'GET':
            return (httplib.OK, self.fixtures.load('ex_list_services.json'), {}, httplib.responses[httplib.OK])
        else:
            return (httplib.OK, self.fixtures.load('ex_deploy_service.json'), {}, httplib.responses[httplib.OK])

    def _v1_services_1s13(self, method, url, body, headers):
        if method == 'GET':
            return (httplib.OK, self.fixtures.load('ex_deploy_service.json'), {}, httplib.responses[httplib.OK])
        elif method == 'DELETE':
            return (httplib.OK, self.fixtures.load('ex_destroy_service.json'), {}, httplib.responses[httplib.OK])

    def _v1_services_1s6(self, method, url, body, headers):
        return (httplib.OK, self.fixtures.load('ex_activate_service.json'), {}, httplib.responses[httplib.OK])

    def _v1_containers(self, method, url, body, headers):
        if '?state=running' in url:
            return (httplib.OK, self.fixtures.load('ex_search_containers.json'), {}, httplib.responses[httplib.OK])
        elif method == 'POST':
            return (httplib.OK, self.fixtures.load('deploy_container.json'), {}, httplib.responses[httplib.OK])
        return (httplib.OK, self.fixtures.load('list_containers.json'), {}, httplib.responses[httplib.OK])

    def _v1_containers_1i31(self, method, url, body, headers):
        if method == 'GET':
            return (httplib.OK, self.fixtures.load('deploy_container.json'), {}, httplib.responses[httplib.OK])
        elif method == 'DELETE' or '?action=stop' in url:
            return (httplib.OK, self.fixtures.load('stop_container.json'), {}, httplib.responses[httplib.OK])
        elif '?action=start' in url:
            return (httplib.OK, self.fixtures.load('start_container.json'), {}, httplib.responses[httplib.OK])
        else:
            return (httplib.OK, self.fixtures.load('deploy_container.json'), {}, httplib.responses[httplib.OK])