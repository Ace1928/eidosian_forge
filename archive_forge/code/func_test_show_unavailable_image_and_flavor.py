import argparse
import base64
import builtins
import collections
import datetime
import io
import os
from unittest import mock
import fixtures
from oslo_utils import timeutils
import testtools
import novaclient
from novaclient import api_versions
from novaclient import base
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
import novaclient.v2.shell
def test_show_unavailable_image_and_flavor(self):
    output, _ = self.run_command('show 9013')
    self.assert_called('GET', '/servers/9013', pos=-6)
    self.assert_called('GET', '/flavors/80645cf4-6ad3-410a-bbc8-6f3e1e291f51', pos=-5)
    self.assert_called('GET', '/v2/images/3e861307-73a6-4d1f-8d68-f68b03223032', pos=-1)
    self.assertIn('Image not found', output)
    self.assertIn('Flavor not found', output)