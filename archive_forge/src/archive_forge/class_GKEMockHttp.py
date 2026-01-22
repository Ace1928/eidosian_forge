import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import GKE_PARAMS, GKE_KEYWORD_PARAMS
from libcloud.common.google import GoogleBaseAuthConnection
from libcloud.test.container import TestCaseMixin
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.gke import API_VERSION, GKEContainerDriver
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
class GKEMockHttp(MockHttp):
    fixtures = ContainerFileFixtures('gke')
    json_hdr = {'content-type': 'application/json; charset=UTF-8'}

    def _get_method_name(self, type, use_param, qs, path):
        api_path = '/%s' % API_VERSION
        project_path = '/projects/%s' % GKE_KEYWORD_PARAMS['project']
        path = path.replace(api_path, '')
        path = path.replace(project_path, '')
        if not path:
            path = '/project'
        method_name = super()._get_method_name(type, use_param, qs, path)
        return method_name

    def _zones_us_central1_a_serverconfig(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_instance_serverconfig.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_clusters(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_list.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])