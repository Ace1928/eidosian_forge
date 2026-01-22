from tests.compat import unittest
from tests.integration import ServiceCertVerificationTest
import boto.logs
class CloudWatchLogsCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    regions = boto.logs.regions()

    def sample_service_call(self, conn):
        conn.describe_log_groups()