from tests.integration import ServiceCertVerificationTest
import boto.ec2.cloudwatch
from tests.compat import unittest
class CloudWatchCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    cloudwatch = True
    regions = boto.ec2.cloudwatch.regions()

    def sample_service_call(self, conn):
        conn.describe_alarms()