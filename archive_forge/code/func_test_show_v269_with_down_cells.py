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
def test_show_v269_with_down_cells(self):
    stdout, _stderr = self.run_command('show 9015', api_version='2.69')
    self.assertEqual('+-----------------------------+---------------------------------------------------+\n| Property                    | Value                                             |\n+-----------------------------+---------------------------------------------------+\n| OS-EXT-AZ:availability_zone | geneva                                            |\n| OS-EXT-STS:power_state      | 0                                                 |\n| created                     | 2018-12-03T21:06:18Z                              |\n| flavor:disk                 | 1                                                 |\n| flavor:ephemeral            | 0                                                 |\n| flavor:extra_specs          | {}                                                |\n| flavor:original_name        | m1.tiny                                           |\n| flavor:ram                  | 512                                               |\n| flavor:swap                 | 0                                                 |\n| flavor:vcpus                | 1                                                 |\n| id                          | 9015                                              |\n| image                       | CentOS 5.2 (c99d7632-bd66-4be9-aed5-3dd14b223a76) |\n| status                      | UNKNOWN                                           |\n| tenant_id                   | 6f70656e737461636b20342065766572                  |\n| user_id                     | fake                                              |\n+-----------------------------+---------------------------------------------------+\n', stdout)
    FAKE_UUID_2 = 'c99d7632-bd66-4be9-aed5-3dd14b223a76'
    self.assert_called('GET', '/servers?name=9015', pos=0)
    self.assert_called('GET', '/servers?name=9015', pos=1)
    self.assert_called('GET', '/servers/9015', pos=2)
    self.assert_called('GET', '/v2/images/%s' % FAKE_UUID_2, pos=3)