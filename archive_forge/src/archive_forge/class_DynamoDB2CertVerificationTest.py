import unittest
from tests.integration import ServiceCertVerificationTest
import boto.dynamodb2
class DynamoDB2CertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    dynamodb2 = True
    regions = boto.dynamodb2.regions()

    def sample_service_call(self, conn):
        conn.list_tables()