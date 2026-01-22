import sys
import os.path
import unittest
import libcloud.pricing
class PricingTestCase(unittest.TestCase):

    def setUp(self):
        super().setUp()
        libcloud.pricing.PRICING_DATA = {'compute': {}, 'storage': {}}

    def test_get_pricing_success(self):
        self.assertFalse('foo' in libcloud.pricing.PRICING_DATA['compute'])
        pricing = libcloud.pricing.get_pricing(driver_type='compute', driver_name='foo', pricing_file_path=PRICING_FILE_PATH)
        self.assertEqual(pricing['1'], 1.0)
        self.assertEqual(pricing['2'], 2.0)
        self.assertEqual(libcloud.pricing.PRICING_DATA['compute']['foo']['1'], 1.0)
        self.assertEqual(libcloud.pricing.PRICING_DATA['compute']['foo']['2'], 2.0)

    def test_get_pricing_invalid_file_path(self):
        try:
            libcloud.pricing.get_pricing(driver_type='compute', driver_name='bar', pricing_file_path='inexistent.json')
        except OSError:
            pass
        else:
            self.fail('Invalid pricing file path provided, but an exception was not thrown')

    def test_get_pricing_invalid_driver_type(self):
        try:
            libcloud.pricing.get_pricing(driver_type='invalid_type', driver_name='bar', pricing_file_path='inexistent.json')
        except AttributeError:
            pass
        else:
            self.fail('Invalid driver_type provided, but an exception was not thrown')

    def test_get_pricing_not_in_cache(self):
        try:
            libcloud.pricing.get_pricing(driver_type='compute', driver_name='inexistent', pricing_file_path=PRICING_FILE_PATH)
        except KeyError:
            pass
        else:
            self.fail('Invalid driver provided, but an exception was not thrown')

    def test_get_size_price(self):
        libcloud.pricing.PRICING_DATA['compute']['foo'] = {2: 2, '3': 3}
        price1 = libcloud.pricing.get_size_price(driver_type='compute', driver_name='foo', size_id=2)
        price2 = libcloud.pricing.get_size_price(driver_type='compute', driver_name='foo', size_id='3')
        self.assertEqual(price1, 2)
        self.assertEqual(price2, 3)

    def test_invalid_pricing_cache(self):
        libcloud.pricing.PRICING_DATA['compute']['foo'] = {2: 2}
        self.assertTrue('foo' in libcloud.pricing.PRICING_DATA['compute'])
        libcloud.pricing.invalidate_pricing_cache()
        self.assertFalse('foo' in libcloud.pricing.PRICING_DATA['compute'])

    def test_invalid_module_pricing_cache(self):
        libcloud.pricing.PRICING_DATA['compute']['foo'] = {1: 1}
        self.assertTrue('foo' in libcloud.pricing.PRICING_DATA['compute'])
        libcloud.pricing.invalidate_module_pricing_cache(driver_type='compute', driver_name='foo')
        self.assertFalse('foo' in libcloud.pricing.PRICING_DATA['compute'])
        libcloud.pricing.invalidate_module_pricing_cache(driver_type='compute', driver_name='foo1')

    def test_set_pricing(self):
        self.assertFalse('foo' in libcloud.pricing.PRICING_DATA['compute'])
        libcloud.pricing.set_pricing(driver_type='compute', driver_name='foo', pricing={'foo': 1})
        self.assertTrue('foo' in libcloud.pricing.PRICING_DATA['compute'])

    def test_get_pricing_data_caching(self):
        self.assertEqual(libcloud.pricing.PRICING_DATA['compute'], {})
        self.assertEqual(libcloud.pricing.PRICING_DATA['storage'], {})
        pricing = libcloud.pricing.get_pricing(driver_type='compute', driver_name='foo', pricing_file_path=PRICING_FILE_PATH)
        self.assertEqual(pricing['1'], 1.0)
        self.assertEqual(pricing['2'], 2.0)
        self.assertEqual(len(libcloud.pricing.PRICING_DATA['compute']), 1)
        self.assertTrue('foo' in libcloud.pricing.PRICING_DATA['compute'])
        pricing = libcloud.pricing.get_pricing(driver_type='compute', driver_name='baz', pricing_file_path=PRICING_FILE_PATH)
        self.assertEqual(pricing['1'], 5.0)
        self.assertEqual(pricing['2'], 6.0)
        self.assertEqual(len(libcloud.pricing.PRICING_DATA['compute']), 2)
        self.assertTrue('foo' in libcloud.pricing.PRICING_DATA['compute'])
        self.assertTrue('baz' in libcloud.pricing.PRICING_DATA['compute'])

    def test_get_pricing_data_module_level_variable_is_true(self):
        self.assertEqual(libcloud.pricing.PRICING_DATA['compute'], {})
        self.assertEqual(libcloud.pricing.PRICING_DATA['storage'], {})
        libcloud.pricing.CACHE_ALL_PRICING_DATA = True
        pricing = libcloud.pricing.get_pricing(driver_type='compute', driver_name='foo', pricing_file_path=PRICING_FILE_PATH, cache_all=False)
        self.assertEqual(pricing['1'], 1.0)
        self.assertEqual(pricing['2'], 2.0)
        self.assertEqual(len(libcloud.pricing.PRICING_DATA['compute']), 3)
        self.assertTrue('foo' in libcloud.pricing.PRICING_DATA['compute'])
        self.assertTrue('bar' in libcloud.pricing.PRICING_DATA['compute'])
        self.assertTrue('baz' in libcloud.pricing.PRICING_DATA['compute'])

    def test_get_pricing_data_caching_cache_all(self):
        self.assertEqual(libcloud.pricing.PRICING_DATA['compute'], {})
        self.assertEqual(libcloud.pricing.PRICING_DATA['storage'], {})
        pricing = libcloud.pricing.get_pricing(driver_type='compute', driver_name='foo', pricing_file_path=PRICING_FILE_PATH, cache_all=True)
        self.assertEqual(pricing['1'], 1.0)
        self.assertEqual(pricing['2'], 2.0)
        self.assertEqual(len(libcloud.pricing.PRICING_DATA['compute']), 3)
        self.assertTrue('foo' in libcloud.pricing.PRICING_DATA['compute'])
        self.assertTrue('bar' in libcloud.pricing.PRICING_DATA['compute'])
        self.assertTrue('baz' in libcloud.pricing.PRICING_DATA['compute'])

    def test_get_gce_image_price_non_premium_image(self):
        image_name = 'debian-10-buster-v20220519'
        cores = 4
        size_name = 'c2d-standard-4'
        price = libcloud.pricing.get_image_price('gce_images', image_name, size_name, cores)
        self.assertTrue(price == 0)

    def test_get_gce_image_price_RHEL_image(self):
        cores = 2
        image_name = 'rhel-7-v20220519'
        size_name = 'n2d-highcpu-2'
        prices = libcloud.pricing.get_pricing('compute', 'gce_images')
        correct_price = float(prices['RHEL']['4vcpu or less']['price'])
        fetched_price = libcloud.pricing.get_image_price('gce_images', image_name, size_name, cores)
        self.assertTrue(fetched_price == correct_price)

    def test_get_gce_image_price_Windows_image(self):
        cores = 2
        image_name = 'windows-server-2012-r2-dc-core-v20220513'
        size_name = 'n2d-highcpu-2'
        prices = libcloud.pricing.get_pricing('compute', 'gce_images')
        correct_price = float(prices['Windows Server']['any']['price']) * 2
        fetched_price = libcloud.pricing.get_image_price('gce_images', image_name, size_name, cores)
        self.assertTrue(fetched_price == correct_price)

    def test_get_gce_image_price_SLES_SAP_image(self):
        cores = 2
        image_name = 'sles-15-sap-v20220126'
        size_name = 'n2d-highcpu-2'
        prices = libcloud.pricing.get_pricing('compute', 'gce_images')
        correct_price = float(prices['SLES for SAP']['1-2vcpu']['price'])
        fetched_price = libcloud.pricing.get_image_price('gce_images', image_name, size_name, cores)
        self.assertTrue(fetched_price == correct_price)

    def test_get_gce_image_price_SQL_image(self):
        image_name = 'sql-2012-standard-windows-2012-r2-dc-v20220513'
        size_name = 'g1 small'
        prices = libcloud.pricing.get_pricing('compute', 'gce_images')
        correct_price = float(prices['SQL Server']['standard']['price'])
        fetched_price = libcloud.pricing.get_image_price('gce_images', image_name, size_name)
        self.assertTrue(fetched_price == correct_price)