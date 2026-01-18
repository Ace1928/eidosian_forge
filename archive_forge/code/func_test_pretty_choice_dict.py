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
def test_pretty_choice_dict(self):
    d = {}
    r = utils.pretty_choice_dict(d)
    self.assertEqual('', r)
    d = {'k1': 'v1', 'k2': 'v2', 'k3': 'v3'}
    r = utils.pretty_choice_dict(d)
    self.assertEqual("'k1=v1', 'k2=v2', 'k3=v3'", r)