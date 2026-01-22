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
class PrintDictTestCase(test_utils.TestCase):

    def test__pretty_format_dict(self):
        content = {'key1': 'value1', 'key2': 'value2'}
        expected = 'key1 : value1\nkey2 : value2'
        result = shell_utils._pretty_format_dict(content)
        self.assertEqual(expected, result)

    def test_print_dict_with_return(self):
        d = {'a': 'A', 'b': 'B', 'c': 'C', 'd': 'test\rcarriage\n\rreturn'}
        with CaptureStdout() as cso:
            shell_utils.print_dict(d)
        self.assertEqual('+----------+---------------+\n| Property | Value         |\n+----------+---------------+\n| a        | A             |\n| b        | B             |\n| c        | C             |\n| d        | test carriage |\n|          |  return       |\n+----------+---------------+\n', cso.read())

    def test_print_dict_with_dict_inside(self):
        content = {'a': 'A', 'b': 'B', 'f_key': {'key1': 'value1', 'key2': 'value2'}}
        with CaptureStdout() as cso:
            shell_utils.print_dict(content, formatters='f_key')
        self.assertEqual('+----------+---------------+\n| Property | Value         |\n+----------+---------------+\n| a        | A             |\n| b        | B             |\n| f_key    | key1 : value1 |\n|          | key2 : value2 |\n+----------+---------------+\n', cso.read())