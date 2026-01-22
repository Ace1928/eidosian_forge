from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
class TestBranchOptions(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.branch = self.make_branch('.')
        self.config_stack = self.branch.get_config_stack()

    def check_append_revisions_only(self, expected_value, value=None):
        """Set append_revisions_only in config and check its interpretation."""
        if value is not None:
            self.config_stack.set('append_revisions_only', value)
        self.assertEqual(expected_value, self.branch.get_append_revisions_only())

    def test_valid_append_revisions_only(self):
        self.assertEqual(None, self.config_stack.get('append_revisions_only'))
        self.check_append_revisions_only(None)
        self.check_append_revisions_only(False, 'False')
        self.check_append_revisions_only(True, 'True')
        self.check_append_revisions_only(False, 'false')
        self.check_append_revisions_only(True, 'true')

    def test_invalid_append_revisions_only(self):
        """Ensure warning is noted on invalid settings"""
        self.warnings = []

        def warning(*args):
            self.warnings.append(args[0] % args[1:])
        self.overrideAttr(trace, 'warning', warning)
        self.check_append_revisions_only(None, 'not-a-bool')
        self.assertLength(1, self.warnings)
        self.assertEqual('Value "not-a-bool" is not valid for "append_revisions_only"', self.warnings[0])

    def test_use_fresh_values(self):
        copy = _mod_branch.Branch.open(self.branch.base)
        copy.lock_write()
        try:
            copy.get_config_stack().set('foo', 'bar')
        finally:
            copy.unlock()
        self.assertFalse(self.branch.is_locked())
        self.assertEqual(None, self.branch.get_config_stack().get('foo'))
        fresh = _mod_branch.Branch.open(self.branch.base)
        self.assertEqual('bar', fresh.get_config_stack().get('foo'))

    def test_set_from_config_get_from_config_stack(self):
        self.branch.lock_write()
        self.addCleanup(self.branch.unlock)
        self.branch.get_config().set_user_option('foo', 'bar')
        result = self.branch.get_config_stack().get('foo')
        self.expectFailure('BranchStack uses cache after set_user_option', self.assertEqual, 'bar', result)

    def test_set_from_config_stack_get_from_config(self):
        self.branch.lock_write()
        self.addCleanup(self.branch.unlock)
        self.branch.get_config_stack().set('foo', 'bar')
        self.assertEqual(None, self.branch.get_config().get_user_option('foo'))

    def test_set_delays_write_when_branch_is_locked(self):
        self.branch.lock_write()
        self.addCleanup(self.branch.unlock)
        self.branch.get_config_stack().set('foo', 'bar')
        copy = _mod_branch.Branch.open(self.branch.base)
        result = copy.get_config_stack().get('foo')
        self.assertIs(None, result)