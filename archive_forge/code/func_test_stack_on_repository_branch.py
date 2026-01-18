from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_stack_on_repository_branch(self):
    try:
        repo = self.make_repository('repo', shared=True)
    except errors.IncompatibleFormat:
        raise TestNotApplicable()
    if not repo._format.supports_nesting_repositories:
        raise TestNotApplicable()
    bzrdir = self.make_controldir('repo/stack-on')
    try:
        b = bzrdir.create_branch()
    except errors.UninitializableFormat:
        raise TestNotApplicable()
    transport = self.get_transport('stacked')
    b.controldir.clone_on_transport(transport, stacked_on=b.base)
    _mod_branch.Branch.open(transport.base)