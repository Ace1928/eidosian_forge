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
def test_pass_through(self):
    value = 42
    for key in ['timeout', 'session', 'original_ip', 'user_agent']:
        args = {key: value}
        self.assertEqual(getattr(self._s(args), key), value)
        self.assertNotIn(key, args)