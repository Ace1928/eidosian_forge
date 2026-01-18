import os
import sys
import threading
from io import BytesIO
from textwrap import dedent
import configobj
from testtools import matchers
from .. import (bedding, branch, config, controldir, diff, errors, lock,
from .. import registry as _mod_registry
from .. import tests, trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..bzr import remote
from ..transport import remote as transport_remote
from . import features, scenarios, test_server
def test_set_user_setting_sets_and_saves2(self):
    self.get_branch_config('/a/c')
    self.assertIs(self.my_config.get_user_option('foo'), None)
    self.my_config.set_user_option('foo', 'bar')
    self.assertEqual(self.my_config.branch.control_files.files['branch.conf'].strip(), b'foo = bar')
    self.assertEqual(self.my_config.get_user_option('foo'), 'bar')
    self.my_config.set_user_option('foo', 'baz', store=config.STORE_LOCATION)
    self.assertEqual(self.my_config.get_user_option('foo'), 'baz')
    self.my_config.set_user_option('foo', 'qux')
    self.assertEqual(self.my_config.get_user_option('foo'), 'baz')