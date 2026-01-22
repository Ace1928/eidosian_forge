import unittest
from tests.integration import ServiceCertVerificationTest
import boto.sts
class STSCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    sts = True
    regions = boto.sts.regions()

    def sample_service_call(self, conn):
        conn.get_session_token()