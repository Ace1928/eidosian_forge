import mock
import os
import json
from nose.tools import assert_equal
from tests.unit import unittest
import boto
from boto.endpoints import BotoEndpointResolver
from boto.endpoints import StaticEndpointBuilder
class EndpointTestCase(object):

    def __init__(self, service, old_endpoints, new_endpoints):
        self.service = service
        self.old_endpoints = old_endpoints
        self.new_endpoints = new_endpoints

    def run(self):
        assert_equal(self.old_endpoints, self.new_endpoints)