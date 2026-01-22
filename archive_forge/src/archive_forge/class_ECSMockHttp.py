import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_ECS
from libcloud.container.base import Container, ContainerImage, ContainerCluster
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.ecs import ElasticContainerDriver
from libcloud.container.utils.docker import RegistryClient
class ECSMockHttp(MockHttp):
    fixtures = ContainerFileFixtures('ecs')
    fixture_map = {'DescribeClusters': 'describeclusters.json', 'CreateCluster': 'createcluster.json', 'DeleteCluster': 'deletecluster.json', 'DescribeTasks': 'describetasks.json', 'ListTasks': 'listtasks.json', 'ListClusters': 'listclusters.json', 'RegisterTaskDefinition': 'registertaskdefinition.json', 'RunTask': 'runtask.json', 'StopTask': 'stoptask.json', 'ListImages': 'listimages.json', 'DescribeRepositories': 'describerepositories.json', 'CreateService': 'createservice.json', 'ListServices': 'listservices.json', 'DescribeServices': 'describeservices.json', 'DeleteService': 'deleteservice.json', 'GetAuthorizationToken': 'getauthorizationtoken.json'}

    def root(self, method, url, body, headers):
        target = headers['x-amz-target']
        if '%s' in self.host:
            self.host = self.host % 'region'
        if target is not None:
            type = target.split('.')[-1]
            if type is None or self.fixture_map.get(type) is None:
                raise AssertionError('Unsupported request type %s' % target)
            body = self.fixtures.load(self.fixture_map.get(type))
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])