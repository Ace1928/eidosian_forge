import http.client as http
import os
import socket
import time
from oslo_serialization import jsonutils
from oslo_utils.fixture import uuidsentinel as uuids
import requests
from glance.common import wsgi
from glance.tests import functional
class TestStagingCleanupSingleStore(functional.FunctionalTest, StagingCleanupBase):
    """Test for staging store cleanup on API server startup.

    This tests the single store case.
    """

    def setUp(self):
        super(TestStagingCleanupSingleStore, self).setUp()
        self.my_api_server = self.api_server
        self._configure_api_server()