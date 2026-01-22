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
class GKEContainerDriverTestCase(GoogleTestCase, TestCaseMixin):
    """
    Google Compute Engine Test Class.
    """
    datacenter = 'us-central1-a'

    def setUp(self):
        GKEMockHttp.test = self
        GKEContainerDriver.connectionCls.conn_class = GKEMockHttp
        GoogleBaseAuthConnection.conn_class = GoogleAuthMockHttp
        GKEMockHttp.type = None
        kwargs = GKE_KEYWORD_PARAMS.copy()
        kwargs['auth_type'] = 'IA'
        kwargs['datacenter'] = self.datacenter
        self.driver = GKEContainerDriver(*GKE_PARAMS, **kwargs)

    def test_list_images_response(self):
        config = self.driver.list_clusters(ex_zone='us-central1-a')
        assert 'clusters' in config
        assert config['clusters'][0]['zone'] == 'us-central1-a'

    def test_server_config(self):
        config = self.driver.get_server_config()
        assert 'validImageTypes' in config