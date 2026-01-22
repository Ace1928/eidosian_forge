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
class TestBranchConfigItems(tests.TestCaseInTempDir):

    def get_branch_config(self, global_config=None, location=None, location_config=None, branch_data_config=None):
        my_branch = FakeBranch(location)
        if global_config is not None:
            config.GlobalConfig.from_string(global_config, save=True)
        if location_config is not None:
            config.LocationConfig.from_string(location_config, my_branch.base, save=True)
        my_config = config.BranchConfig(my_branch)
        if branch_data_config is not None:
            my_config.branch.control_files.files['branch.conf'] = branch_data_config
        return my_config

    def test_user_id(self):
        branch = FakeBranch()
        my_config = config.BranchConfig(branch)
        self.assertIsNot(None, my_config.username())
        my_config.branch.control_files.files['email'] = 'John'
        my_config.set_user_option('email', 'Robert Collins <robertc@example.org>')
        self.assertEqual('Robert Collins <robertc@example.org>', my_config.username())

    def test_BRZ_EMAIL_OVERRIDES(self):
        self.overrideEnv('BRZ_EMAIL', 'Robert Collins <robertc@example.org>')
        branch = FakeBranch()
        my_config = config.BranchConfig(branch)
        self.assertEqual('Robert Collins <robertc@example.org>', my_config.username())

    def test_get_user_option_global(self):
        my_config = self.get_branch_config(global_config=sample_config_text)
        self.assertEqual('something', my_config.get_user_option('user_global_option'))

    def test_config_precedence(self):
        my_config = self.get_branch_config(global_config=precedence_global)
        self.assertEqual(my_config.get_user_option('option'), 'global')
        my_config = self.get_branch_config(global_config=precedence_global, branch_data_config=precedence_branch)
        self.assertEqual(my_config.get_user_option('option'), 'branch')
        my_config = self.get_branch_config(global_config=precedence_global, branch_data_config=precedence_branch, location_config=precedence_location)
        self.assertEqual(my_config.get_user_option('option'), 'recurse')
        my_config = self.get_branch_config(global_config=precedence_global, branch_data_config=precedence_branch, location_config=precedence_location, location='http://example.com/specific')
        self.assertEqual(my_config.get_user_option('option'), 'exact')