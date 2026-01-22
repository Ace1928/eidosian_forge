from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
class TestBranchFormat5(tests.TestCaseWithTransport):
    """Tests specific to branch format 5"""

    def test_branch_format_5_uses_lockdir(self):
        url = self.get_url()
        bdir = bzrdir.BzrDirMetaFormat1().initialize(url)
        bdir.create_repository()
        branch = BzrBranchFormat5().initialize(bdir)
        t = self.get_transport()
        self.log('branch instance is %r' % branch)
        self.assertTrue(isinstance(branch, BzrBranch5))
        self.assertIsDirectory('.', t)
        self.assertIsDirectory('.bzr/branch', t)
        self.assertIsDirectory('.bzr/branch/lock', t)
        branch.lock_write()
        self.addCleanup(branch.unlock)
        self.assertIsDirectory('.bzr/branch/lock/held', t)

    def test_set_push_location(self):
        conf = config.LocationConfig.from_string('# comment\n', '.', save=True)
        branch = self.make_branch('.', format='knit')
        branch.set_push_location('foo')
        local_path = urlutils.local_path_from_url(branch.base[:-1])
        self.assertFileEqual(b'# comment\n[%s]\npush_location = foo\npush_location:policy = norecurse\n' % local_path.encode('utf-8'), bedding.locations_config_path())