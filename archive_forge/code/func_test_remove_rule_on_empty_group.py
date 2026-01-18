from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.connection import EC2Connection
from boto.ec2.securitygroup import SecurityGroup
def test_remove_rule_on_empty_group(self):
    sg = SecurityGroup()
    with self.assertRaises(ValueError):
        sg.remove_rule('ip', 80, 80, None, None, None, None)