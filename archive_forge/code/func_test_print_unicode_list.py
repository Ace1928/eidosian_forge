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
def test_print_unicode_list(self):
    objs = [_FakeResult('k', '…')]
    utils.print_list(objs, ['Name', 'Value'])
    s = '…'
    self.assertEqual('+------+-------+\n| Name | Value |\n+------+-------+\n| k    | %s     |\n+------+-------+\n' % s, sys.stdout.getvalue())