import boto
from boto.awslambda.exceptions import ResourceNotFoundException
from tests.compat import unittest
def test_resource_not_found_exceptions(self):
    with self.assertRaises(ResourceNotFoundException):
        self.awslambda.get_function(function_name='non-existant-function')