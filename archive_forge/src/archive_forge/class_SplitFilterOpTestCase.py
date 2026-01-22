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
class SplitFilterOpTestCase(test_utils.BaseTestCase):

    def test_less_than_operator(self):
        expr = 'lt:bar'
        returned = utils.split_filter_op(expr)
        self.assertEqual(('lt', 'bar'), returned)

    def test_less_than_equal_operator(self):
        expr = 'lte:bar'
        returned = utils.split_filter_op(expr)
        self.assertEqual(('lte', 'bar'), returned)

    def test_greater_than_operator(self):
        expr = 'gt:bar'
        returned = utils.split_filter_op(expr)
        self.assertEqual(('gt', 'bar'), returned)

    def test_greater_than_equal_operator(self):
        expr = 'gte:bar'
        returned = utils.split_filter_op(expr)
        self.assertEqual(('gte', 'bar'), returned)

    def test_not_equal_operator(self):
        expr = 'neq:bar'
        returned = utils.split_filter_op(expr)
        self.assertEqual(('neq', 'bar'), returned)

    def test_equal_operator(self):
        expr = 'eq:bar'
        returned = utils.split_filter_op(expr)
        self.assertEqual(('eq', 'bar'), returned)

    def test_in_operator(self):
        expr = 'in:bar'
        returned = utils.split_filter_op(expr)
        self.assertEqual(('in', 'bar'), returned)

    def test_split_filter_value_for_quotes(self):
        expr = '"fake\\"name",fakename,"fake,name"'
        returned = utils.split_filter_value_for_quotes(expr)
        list_values = ['fake\\"name', 'fakename', 'fake,name']
        self.assertEqual(list_values, returned)

    def test_validate_quotes(self):
        expr = '"aaa\\"aa",bb,"cc"'
        returned = utils.validate_quotes(expr)
        self.assertIsNone(returned)
        invalid_expr = ['"aa', 'ss"', 'aa"bb"cc', '"aa""bb"']
        for expr in invalid_expr:
            self.assertRaises(exception.InvalidParameterValue, utils.validate_quotes, expr)

    def test_default_operator(self):
        expr = 'bar'
        returned = utils.split_filter_op(expr)
        self.assertEqual(('eq', expr), returned)

    def test_default_operator_with_datetime(self):
        expr = '2015-08-27T09:49:58Z'
        returned = utils.split_filter_op(expr)
        self.assertEqual(('eq', expr), returned)

    def test_operator_with_datetime(self):
        expr = 'lt:2015-08-27T09:49:58Z'
        returned = utils.split_filter_op(expr)
        self.assertEqual(('lt', '2015-08-27T09:49:58Z'), returned)