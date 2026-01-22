import argparse
from io import StringIO
import itertools
import logging
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_config import fixture as config
from oslo_serialization import jsonutils
import requests
from testtools import matchers
from keystoneclient import adapter
from keystoneclient.auth import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
from keystoneclient import session as client_session
from keystoneclient.tests.unit import utils
class ConstructSessionFromArgsTests(utils.TestCase):
    KEY = 'keyfile'
    CERT = 'certfile'
    CACERT = 'cacert-path'

    def _s(self, k=None, **kwargs):
        k = k or kwargs
        with self.deprecations.expect_deprecations_here():
            return client_session.Session.construct(k)

    def test_verify(self):
        self.assertFalse(self._s(insecure=True).verify)
        self.assertTrue(self._s(verify=True, insecure=True).verify)
        self.assertFalse(self._s(verify=False, insecure=True).verify)
        self.assertEqual(self._s(cacert=self.CACERT).verify, self.CACERT)

    def test_cert(self):
        tup = (self.CERT, self.KEY)
        self.assertEqual(self._s(cert=tup).cert, tup)
        self.assertEqual(self._s(cert=self.CERT, key=self.KEY).cert, tup)
        self.assertIsNone(self._s(key=self.KEY).cert)

    def test_pass_through(self):
        value = 42
        for key in ['timeout', 'session', 'original_ip', 'user_agent']:
            args = {key: value}
            self.assertEqual(getattr(self._s(args), key), value)
            self.assertNotIn(key, args)