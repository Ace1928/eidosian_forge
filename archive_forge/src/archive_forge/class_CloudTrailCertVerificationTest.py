import unittest
from tests.integration import ServiceCertVerificationTest
import boto.cloudtrail
class CloudTrailCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    cloudtrail = True
    regions = boto.cloudtrail.regions()

    def sample_service_call(self, conn):
        conn.describe_trails()