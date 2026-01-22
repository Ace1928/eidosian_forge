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
class TestConfigGetOptions(tests.TestCaseWithTransport, TestOptionsMixin):

    def setUp(self):
        super().setUp()
        create_configs(self)

    def test_no_variable(self):
        self.assertOptions([], self.branch_config)

    def test_option_in_breezy(self):
        self.breezy_config.set_user_option('file', 'breezy')
        self.assertOptions([('file', 'breezy', 'DEFAULT', 'breezy')], self.breezy_config)

    def test_option_in_locations(self):
        self.locations_config.set_user_option('file', 'locations')
        self.assertOptions([('file', 'locations', self.tree.basedir, 'locations')], self.locations_config)

    def test_option_in_branch(self):
        self.branch_config.set_user_option('file', 'branch')
        self.assertOptions([('file', 'branch', 'DEFAULT', 'branch')], self.branch_config)

    def test_option_in_breezy_and_branch(self):
        self.breezy_config.set_user_option('file', 'breezy')
        self.branch_config.set_user_option('file', 'branch')
        self.assertOptions([('file', 'branch', 'DEFAULT', 'branch'), ('file', 'breezy', 'DEFAULT', 'breezy')], self.branch_config)

    def test_option_in_branch_and_locations(self):
        self.locations_config.set_user_option('file', 'locations')
        self.branch_config.set_user_option('file', 'branch')
        self.assertOptions([('file', 'locations', self.tree.basedir, 'locations'), ('file', 'branch', 'DEFAULT', 'branch')], self.branch_config)

    def test_option_in_breezy_locations_and_branch(self):
        self.breezy_config.set_user_option('file', 'breezy')
        self.locations_config.set_user_option('file', 'locations')
        self.branch_config.set_user_option('file', 'branch')
        self.assertOptions([('file', 'locations', self.tree.basedir, 'locations'), ('file', 'branch', 'DEFAULT', 'branch'), ('file', 'breezy', 'DEFAULT', 'breezy')], self.branch_config)