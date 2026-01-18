from io import BytesIO
from testtools.matchers import Equals, MatchesAny
from ... import branch, check, controldir, errors, push, tests
from ...branch import BindingUnsupported, Branch
from ...bzr import branch as bzrbranch
from ...bzr import vf_repository
from ...bzr.smart.repository import SmartServerRepositoryGetParentMap
from ...controldir import ControlDir
from ...revision import NULL_REVISION
from .. import test_server
from . import TestCaseWithInterBranch
def test_post_push_nonempty_history(self):
    target = self.make_to_branch_and_tree('target')
    target.lock_write()
    target.add('')
    rev1 = target.commit('rev 1')
    target.unlock()
    sourcedir = target.branch.controldir.clone(self.get_url('source'))
    source = sourcedir.open_branch().create_memorytree()
    rev2 = source.commit('rev 2')
    Branch.hooks.install_named_hook('post_push', self.capture_post_push_hook, None)
    source.branch.push(target.branch)
    self.assertEqual([('post_push', source.branch, None, target.branch.base, 1, rev1, 2, rev2, True, None, True)], self.hook_calls)