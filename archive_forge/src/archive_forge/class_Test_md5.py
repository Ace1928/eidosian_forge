import concurrent.futures
import hashlib
import logging
import sys
from unittest import mock
import fixtures
import os_service_types
import testtools
import openstack
from openstack import exceptions
from openstack.tests.unit import base
from openstack import utils
class Test_md5(base.TestCase):

    def setUp(self):
        super(Test_md5, self).setUp()
        self.md5_test_data = 'Openstack forever'.encode('utf-8')
        try:
            self.md5_digest = hashlib.md5(self.md5_test_data).hexdigest()
            self.fips_enabled = False
        except ValueError:
            self.md5_digest = '0d6dc3c588ae71a04ce9a6beebbbba06'
            self.fips_enabled = True

    def test_md5_with_data(self):
        if not self.fips_enabled:
            digest = utils.md5(self.md5_test_data).hexdigest()
            self.assertEqual(digest, self.md5_digest)
        else:
            self.assertRaises(ValueError, utils.md5, self.md5_test_data)
        if not self.fips_enabled:
            digest = utils.md5(self.md5_test_data, usedforsecurity=True).hexdigest()
            self.assertEqual(digest, self.md5_digest)
        else:
            self.assertRaises(ValueError, utils.md5, self.md5_test_data, usedforsecurity=True)
        digest = utils.md5(self.md5_test_data, usedforsecurity=False).hexdigest()
        self.assertEqual(digest, self.md5_digest)

    def test_md5_without_data(self):
        if not self.fips_enabled:
            test_md5 = utils.md5()
            test_md5.update(self.md5_test_data)
            digest = test_md5.hexdigest()
            self.assertEqual(digest, self.md5_digest)
        else:
            self.assertRaises(ValueError, utils.md5)
        if not self.fips_enabled:
            test_md5 = utils.md5(usedforsecurity=True)
            test_md5.update(self.md5_test_data)
            digest = test_md5.hexdigest()
            self.assertEqual(digest, self.md5_digest)
        else:
            self.assertRaises(ValueError, utils.md5, usedforsecurity=True)
        test_md5 = utils.md5(usedforsecurity=False)
        test_md5.update(self.md5_test_data)
        digest = test_md5.hexdigest()
        self.assertEqual(digest, self.md5_digest)

    def test_string_data_raises_type_error(self):
        if not self.fips_enabled:
            self.assertRaises(TypeError, hashlib.md5, u'foo')
            self.assertRaises(TypeError, utils.md5, u'foo')
            self.assertRaises(TypeError, utils.md5, u'foo', usedforsecurity=True)
        else:
            self.assertRaises(ValueError, hashlib.md5, u'foo')
            self.assertRaises(ValueError, utils.md5, u'foo')
            self.assertRaises(ValueError, utils.md5, u'foo', usedforsecurity=True)
        self.assertRaises(TypeError, utils.md5, u'foo', usedforsecurity=False)

    def test_none_data_raises_type_error(self):
        if not self.fips_enabled:
            self.assertRaises(TypeError, hashlib.md5, None)
            self.assertRaises(TypeError, utils.md5, None)
            self.assertRaises(TypeError, utils.md5, None, usedforsecurity=True)
        else:
            self.assertRaises(ValueError, hashlib.md5, None)
            self.assertRaises(ValueError, utils.md5, None)
            self.assertRaises(ValueError, utils.md5, None, usedforsecurity=True)
        self.assertRaises(TypeError, utils.md5, None, usedforsecurity=False)