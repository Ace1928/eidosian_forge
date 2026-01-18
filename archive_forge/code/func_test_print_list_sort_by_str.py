import io
import sys
from unittest import mock
from urllib import parse
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import fakes
from novaclient.tests.unit import utils as test_utils
from novaclient import utils
from novaclient.v2 import servers
@mock.patch('sys.stdout', io.StringIO())
def test_print_list_sort_by_str(self):
    objs = [_FakeResult('k1', 1), _FakeResult('k3', 2), _FakeResult('k2', 3)]
    utils.print_list(objs, ['Name', 'Value'], sortby_index=0)
    self.assertEqual('+------+-------+\n| Name | Value |\n+------+-------+\n| k1   | 1     |\n| k2   | 3     |\n| k3   | 2     |\n+------+-------+\n', sys.stdout.getvalue())