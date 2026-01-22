import time
from tests.unit import unittest
from boto.cloudsearch.layer1 import Layer1
from boto.cloudsearch.layer2 import Layer2
from boto.regioninfo import RegionInfo
class CloudSearchLayer2Test(unittest.TestCase):
    cloudsearch = True

    def setUp(self):
        super(CloudSearchLayer2Test, self).setUp()
        self.layer2 = Layer2()
        self.domain_name = 'test-%d' % int(time.time())

    def test_create_domain(self):
        domain = self.layer2.create_domain(self.domain_name)
        self.addCleanup(domain.delete)
        self.assertTrue(domain.created, False)
        self.assertEqual(domain.domain_name, self.domain_name)
        self.assertEqual(domain.num_searchable_docs, 0)

    def test_initialization_regression(self):
        us_west_2 = RegionInfo(name='us-west-2', endpoint='cloudsearch.us-west-2.amazonaws.com')
        self.layer2 = Layer2(region=us_west_2, host='cloudsearch.us-west-2.amazonaws.com')
        self.assertEqual(self.layer2.layer1.host, 'cloudsearch.us-west-2.amazonaws.com')