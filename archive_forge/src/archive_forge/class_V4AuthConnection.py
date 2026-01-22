import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
class V4AuthConnection(AWSAuthConnection):

    def __init__(self, host, aws_access_key_id, aws_secret_access_key, port=443):
        AWSAuthConnection.__init__(self, host, aws_access_key_id, aws_secret_access_key, port=port)

    def _required_auth_capability(self):
        return ['hmac-v4']