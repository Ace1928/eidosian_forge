import re
from io import BytesIO
from ... import branch as _mod_branch
from ... import commit, controldir
from ... import delta as _mod_delta
from ... import errors, gpg, info, repository
from ... import revision as _mod_revision
from ... import tests, transport, upgrade, workingtree
from ...bzr import branch as _mod_bzrbranch
from ...bzr import inventory, knitpack_repo, remote
from ...bzr import repository as bzrrepository
from .. import per_repository, test_server
from ..matchers import *
def test_renamed(self):
    self.assertTrue(self.repository.revision_tree(self.rev2).has_filename('newname'))
    self.assertTrue(self.repository.revision_tree(self.rev1).has_filename('oldname'))
    revs = [self.repository.get_revision(self.rev2), self.repository.get_revision(self.rev1)]
    delta2, delta1 = list(self.repository.get_revision_deltas(revs, specific_files=['newname']))
    self.assertIsInstance(delta1, _mod_delta.TreeDelta)
    self.assertEqual([('oldname', 'newname')], [c.path for c in delta2.renamed])
    self.assertIsInstance(delta2, _mod_delta.TreeDelta)
    self.assertEqual(['oldname'], [c.path[1] for c in delta1.added])