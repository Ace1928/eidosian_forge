import mock
import os
import json
from nose.tools import assert_equal
from tests.unit import unittest
import boto
from boto.endpoints import BotoEndpointResolver
from boto.endpoints import StaticEndpointBuilder
class EndpointPreservedTestCase(object):

    def __init__(self, service_name, region_name, old_endpoint, new_endpoint):
        self.service_name = service_name
        self.region_name = region_name
        self.old_endpoint = old_endpoint
        self.new_endpoint = new_endpoint

    def run(self):
        message = 'Endpoint for %s in %s does not match snapshot: %s != %s' % (self.service_name, self.region_name, self.new_endpoint, self.old_endpoint)
        assert_equal(self.old_endpoint, self.new_endpoint, message)