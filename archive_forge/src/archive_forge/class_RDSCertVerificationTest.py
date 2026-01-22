import unittest
from tests.integration import ServiceCertVerificationTest
import boto.rds
class RDSCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    rds = True
    regions = boto.rds.regions()

    def sample_service_call(self, conn):
        conn.get_all_dbinstances()