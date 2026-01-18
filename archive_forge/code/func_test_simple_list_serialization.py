import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
def test_simple_list_serialization(self):
    params = {}
    self.connection.build_list_params(params, ['foo', 'bar', 'baz'], 'ParamName.member')
    self.assertDictEqual({'ParamName.member.1': 'foo', 'ParamName.member.2': 'bar', 'ParamName.member.3': 'baz'}, params)