from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_select_bind_location(self):
    branch = self.make_branch('branch')
    reconfiguration = reconfigure.Reconfigure(branch.controldir)
    self.assertRaises(reconfigure.NoBindLocation, reconfiguration._select_bind_location)
    branch.set_parent('http://parent')
    reconfiguration = reconfigure.Reconfigure(branch.controldir)
    self.assertEqual('http://parent', reconfiguration._select_bind_location())
    branch.set_push_location('sftp://push')
    reconfiguration = reconfigure.Reconfigure(branch.controldir)
    self.assertEqual('sftp://push', reconfiguration._select_bind_location())
    branch.lock_write()
    try:
        branch.set_bound_location('bzr://foo/old-bound')
        branch.set_bound_location(None)
    finally:
        branch.unlock()
    reconfiguration = reconfigure.Reconfigure(branch.controldir)
    self.assertEqual('bzr://foo/old-bound', reconfiguration._select_bind_location())
    branch.set_bound_location('bzr://foo/cur-bound')
    reconfiguration = reconfigure.Reconfigure(branch.controldir)
    self.assertEqual('bzr://foo/cur-bound', reconfiguration._select_bind_location())
    reconfiguration.new_bound_location = 'ftp://user-specified'
    self.assertEqual('ftp://user-specified', reconfiguration._select_bind_location())