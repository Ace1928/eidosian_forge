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
def test_strict_switched_off_in_configparser(self):
    temp_dir = self.useFixture(fixtures.TempDir()).path
    os.mkdir(os.path.join(temp_dir, 'nested'))
    temp_file = os.path.join(temp_dir, 'test.conf')
    f = open(temp_file, 'w')
    f.write('[Filters]\nprivsep: PathFilter, privsep-helper, root\nprivsep: PathFilter, privsep-helper, root\n')
    f.close()
    filterlist = wrapper.load_filters([temp_dir])
    self.assertIsNotNone(filterlist)