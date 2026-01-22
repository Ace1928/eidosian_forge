import logging as std_logging
import os
import os.path
import random
from unittest import mock
import fixtures
from oslo_config import cfg
from oslo_db import options as db_options
from oslo_utils import strutils
import pbr.version
import testtools
from neutron_lib._i18n import _
from neutron_lib import constants
from neutron_lib import exceptions
from neutron_lib import fixture
from neutron_lib.tests import _post_mortem_debug as post_mortem_debug
def assertDictSupersetOf(self, expected_subset, actual_superset):
    """Checks that actual dict contains the expected dict.

        After checking that the arguments are of the right type, this checks
        that each item in expected_subset is in, and matches, what is in
        actual_superset. Separate tests are done, so that detailed info can
        be reported upon failure.
        """
    if not isinstance(expected_subset, dict):
        self.fail('expected_subset (%s) is not an instance of dict' % type(expected_subset))
    if not isinstance(actual_superset, dict):
        self.fail('actual_superset (%s) is not an instance of dict' % type(actual_superset))
    for k, v in expected_subset.items():
        self.assertIn(k, actual_superset)
        self.assertEqual(v, actual_superset[k], 'Key %(key)s expected: %(exp)r, actual %(act)r' % {'key': k, 'exp': v, 'act': actual_superset[k]})