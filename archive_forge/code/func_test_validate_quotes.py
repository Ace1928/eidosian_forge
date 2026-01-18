import io
import tempfile
from unittest import mock
import glance_store as store
from glance_store._drivers import cinder
from oslo_config import cfg
from oslo_log import log as logging
import webob
from glance.common import exception
from glance.common import store_utils
from glance.common import utils
from glance.tests.unit import base
from glance.tests import utils as test_utils
def test_validate_quotes(self):
    expr = '"aaa\\"aa",bb,"cc"'
    returned = utils.validate_quotes(expr)
    self.assertIsNone(returned)
    invalid_expr = ['"aa', 'ss"', 'aa"bb"cc', '"aa""bb"']
    for expr in invalid_expr:
        self.assertRaises(exception.InvalidParameterValue, utils.validate_quotes, expr)