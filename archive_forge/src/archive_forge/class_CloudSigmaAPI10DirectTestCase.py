import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts
from libcloud.compute.base import Node
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import CloudSigmaNodeDriver, CloudSigmaZrhNodeDriver
class CloudSigmaAPI10DirectTestCase(CloudSigmaAPI10BaseTestCase, unittest.TestCase):
    driver_klass = CloudSigmaZrhNodeDriver
    driver_args = ('foo', 'bar')
    driver_kwargs = {}