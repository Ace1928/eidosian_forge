from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_transform_fallback_location_hook(self):
    stack_on = self.make_branch('stack-on')
    stacked = self.make_branch('stacked')
    try:
        stacked.set_stacked_on_url('../stack-on')
    except unstackable_format_errors as e:
        raise TestNotApplicable('Format does not support stacking.')
    self.get_transport().rename('stack-on', 'new-stack-on')
    hook_calls = []

    def hook(stacked_branch, url):
        hook_calls.append(url)
        return '../new-stack-on'
    _mod_branch.Branch.hooks.install_named_hook('transform_fallback_location', hook, None)
    _mod_branch.Branch.open('stacked')
    self.assertEqual(['../stack-on'], hook_calls)