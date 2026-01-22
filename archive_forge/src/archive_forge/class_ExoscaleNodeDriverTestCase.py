import sys
from libcloud.test import unittest
from libcloud.compute.drivers.exoscale import ExoscaleNodeDriver
from libcloud.test.compute.test_cloudstack import CloudStackCommonTestCase
class ExoscaleNodeDriverTestCase(CloudStackCommonTestCase, unittest.TestCase):
    driver_klass = ExoscaleNodeDriver