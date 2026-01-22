import unittest
from tests.integration import ServiceCertVerificationTest
import boto.cloudformation
class CloudFormationCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    cloudformation = True
    regions = boto.cloudformation.regions()

    def sample_service_call(self, conn):
        conn.describe_stacks()