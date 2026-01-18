import boto
from tests.compat import unittest
from boto.cloudhsm.exceptions import InvalidRequestException
def test_validation_exception(self):
    invalid_arn = 'arn:aws:cloudhsm:us-east-1:123456789012:hapg-55214b8d'
    with self.assertRaises(InvalidRequestException):
        self.cloudhsm.describe_hapg(invalid_arn)