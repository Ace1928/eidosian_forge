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
def test_option_in_branch_and_locations(self):
    self.locations_config.set_user_option('file', 'locations')
    self.branch_config.set_user_option('file', 'branch')
    self.assertOptions([('file', 'locations', self.tree.basedir, 'locations'), ('file', 'branch', 'DEFAULT', 'branch')], self.branch_config)