import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_doesnt_create_broken_branch(self):
    """Pushing a new standalone branch works even when there's a default
        stacking policy at the destination.

        The new branch will preserve the repo format (even if it isn't the
        default for the branch), and will be stacked when the repo format
        allows (which means that the branch format isn't necessarly preserved).
        """
    self.make_repository('repo', shared=True, format='1.6')
    builder = self.make_branch_builder('repo/local', format='pack-0.92')
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', '')), ('add', ('filename', b'f-id', 'file', b'content\n'))], revision_id=b'rev-1')
    builder.build_snapshot([b'rev-1'], [], revision_id=b'rev-2')
    builder.build_snapshot([b'rev-2'], [('modify', ('filename', b'new-content\n'))], revision_id=b'rev-3')
    builder.finish_series()
    branch = builder.get_branch()
    self.run_bzr('push -d repo/local trunk -r 1')
    self.make_controldir('.').get_config().set_default_stack_on('trunk')
    out, err = self.run_bzr('push -d repo/local remote -r 2')
    self.assertContainsRe(err, 'Using default stacking branch trunk at .*')
    out, err = self.run_bzr('push -d repo/local remote -r 3')