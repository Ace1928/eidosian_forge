import unittest
from tests.integration import ServiceCertVerificationTest
import boto.sdb
class SDBCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    sdb = True
    regions = boto.sdb.regions()

    def sample_service_call(self, conn):
        conn.get_all_domains()