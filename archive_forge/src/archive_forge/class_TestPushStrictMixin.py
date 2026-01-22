import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
class TestPushStrictMixin:

    def make_local_branch_and_tree(self):
        self.tree = self.make_branch_and_tree('local')
        self.build_tree_contents([('local/file', b'initial')])
        self.tree.add('file')
        self.tree.commit('adding file', rev_id=b'added')
        self.build_tree_contents([('local/file', b'modified')])
        self.tree.commit('modify file', rev_id=b'modified')

    def set_config_push_strict(self, value):
        br = branch.Branch.open('local')
        br.get_config_stack().set('push_strict', value)
    _default_command = ['push', '../to']
    _default_wd = 'local'
    _default_errors = ['Working tree ".*/local/" has uncommitted changes \\(See brz status\\)\\.']
    _default_additional_error = 'Use --no-strict to force the push.\n'
    _default_additional_warning = 'Uncommitted changes will not be pushed.'

    def assertPushFails(self, args):
        out, err = self.run_bzr_error(self._default_errors, self._default_command + args, working_dir=self._default_wd, retcode=3)
        self.assertContainsRe(err, self._default_additional_error)

    def assertPushSucceeds(self, args, with_warning=False, revid_to_push=None):
        if with_warning:
            error_regexes = self._default_errors
        else:
            error_regexes = []
        out, err = self.run_bzr(self._default_command + args, working_dir=self._default_wd, error_regexes=error_regexes)
        if with_warning:
            self.assertContainsRe(err, self._default_additional_warning)
        else:
            self.assertNotContainsRe(err, self._default_additional_warning)
        branch_from = branch.Branch.open(self._default_wd)
        if revid_to_push is None:
            revid_to_push = branch_from.last_revision()
        branch_to = branch.Branch.open('to')
        repo_to = branch_to.repository
        self.assertTrue(repo_to.has_revision(revid_to_push))
        self.assertEqual(revid_to_push, branch_to.last_revision())