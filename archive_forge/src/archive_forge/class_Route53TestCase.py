import time
import unittest
from nose.plugins.attrib import attr
from boto.route53.connection import Route53Connection
@attr(route53=True)
class Route53TestCase(unittest.TestCase):

    def setUp(self):
        super(Route53TestCase, self).setUp()
        self.conn = Route53Connection()
        self.base_domain = 'boto-test-%s.com' % str(int(time.time()))
        self.zone = self.conn.create_zone(self.base_domain)

    def tearDown(self):
        self.zone.delete()
        super(Route53TestCase, self).tearDown()