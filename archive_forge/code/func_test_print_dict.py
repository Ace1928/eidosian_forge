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
def test_print_dict(self):
    dict = {'key': 'value'}
    utils.print_dict(dict)
    self.assertEqual('+----------+-------+\n| Property | Value |\n+----------+-------+\n| key      | value |\n+----------+-------+\n', sys.stdout.getvalue())