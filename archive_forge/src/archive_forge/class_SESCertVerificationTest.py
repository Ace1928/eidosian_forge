import unittest
from tests.integration import ServiceCertVerificationTest
import boto.ses
class SESCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    ses = True
    regions = boto.ses.regions()

    def sample_service_call(self, conn):
        conn.list_verified_email_addresses()