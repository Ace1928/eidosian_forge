import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
class TestAWSQueryConnection(unittest.TestCase):

    def setUp(self):
        self.region = RegionInfo(name='cc-zone-1', endpoint='mockservice.cc-zone-1.amazonaws.com', connection_cls=MockAWSService)
        HTTPretty.enable()

    def tearDown(self):
        HTTPretty.disable()