import configparser
import logging
import logging.handlers
import os
import tempfile
from unittest import mock
import uuid
import fixtures
import testtools
from oslo_rootwrap import cmd
from oslo_rootwrap import daemon
from oslo_rootwrap import filters
from oslo_rootwrap import subprocess
from oslo_rootwrap import wrapper
def _test_EnvFilter_as_DnsMasq(self, config_file_arg):
    usercmd = ['env', config_file_arg + '=A', 'NETWORK_ID=foobar', 'dnsmasq', 'foo']
    f = filters.EnvFilter('env', 'root', config_file_arg + '=A', 'NETWORK_ID=', '/usr/bin/dnsmasq')
    self.assertTrue(f.match(usercmd))
    self.assertEqual(['/usr/bin/dnsmasq', 'foo'], f.get_command(usercmd))
    env = f.get_environment(usercmd)
    self.assertEqual('A', env.get(config_file_arg))
    self.assertEqual('foobar', env.get('NETWORK_ID'))