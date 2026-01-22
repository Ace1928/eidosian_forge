import collections
import io
import sys
from unittest import mock
import ddt
from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient.tests.unit import utils as test_utils
from cinderclient import utils
@ddt.ddt
class BuildQueryParamTestCase(test_utils.TestCase):

    def test_build_param_without_sort_switch(self):
        dict_param = {'key1': 'val1', 'key2': 'val2', 'key3': 'val3'}
        result = utils.build_query_param(dict_param, True)
        self.assertIn('key1=val1', result)
        self.assertIn('key2=val2', result)
        self.assertIn('key3=val3', result)

    def test_build_param_with_sort_switch(self):
        dict_param = {'key1': 'val1', 'key2': 'val2', 'key3': 'val3'}
        result = utils.build_query_param(dict_param, True)
        expected = '?key1=val1&key2=val2&key3=val3'
        self.assertEqual(expected, result)

    @ddt.data({}, None, {'key1': 'val1', 'key2': None, 'key3': False, 'key4': ''})
    def test_build_param_with_nones(self, dict_param):
        result = utils.build_query_param(dict_param)
        expected = ('key1=val1', 'key3=False') if dict_param else ()
        for exp in expected:
            self.assertIn(exp, result)
        if not expected:
            self.assertEqual('', result)