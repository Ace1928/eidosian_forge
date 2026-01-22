from tests.compat import unittest
from nose.plugins.attrib import attr
from tests.integration import ServiceCertVerificationTest
import boto.route53
@attr(route53=True)
class Route53CertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    route53 = True
    regions = boto.route53.regions()

    def sample_service_call(self, conn):
        conn.get_all_hosted_zones()