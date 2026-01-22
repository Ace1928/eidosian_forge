import unittest
from tests.integration import ServiceCertVerificationTest
import boto.support
class SupportCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    support = True
    regions = boto.support.regions()

    def sample_service_call(self, conn):
        conn.describe_services()