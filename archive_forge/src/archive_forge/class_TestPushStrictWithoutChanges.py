import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
class TestPushStrictWithoutChanges(tests.TestCaseWithTransport, TestPushStrictMixin):

    def setUp(self):
        super().setUp()
        self.make_local_branch_and_tree()

    def test_push_default(self):
        self.assertPushSucceeds([])

    def test_push_strict(self):
        self.assertPushSucceeds(['--strict'])

    def test_push_no_strict(self):
        self.assertPushSucceeds(['--no-strict'])

    def test_push_config_var_strict(self):
        self.set_config_push_strict('true')
        self.assertPushSucceeds([])

    def test_push_config_var_no_strict(self):
        self.set_config_push_strict('false')
        self.assertPushSucceeds([])