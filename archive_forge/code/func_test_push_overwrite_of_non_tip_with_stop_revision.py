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
def test_push_overwrite_of_non_tip_with_stop_revision(self):
    """Combining the stop_revision and overwrite options works.

        This was <https://bugs.launchpad.net/bzr/+bug/234229>.
        """
    source = self.make_from_branch_and_tree('source')
    target = self.make_to_branch('target')
    source.commit('1st commit')
    try:
        source.branch.push(target)
    except errors.NoRoundtrippingSupport:
        raise tests.TestNotApplicable('lossless push between %r and %r not supported' % (self.branch_format_from, self.branch_format_to))
    rev2 = source.commit('2nd commit')
    source.commit('3rd commit')
    source.branch.push(target, stop_revision=rev2, overwrite=True)
    self.assertEqual(rev2, target.last_revision())