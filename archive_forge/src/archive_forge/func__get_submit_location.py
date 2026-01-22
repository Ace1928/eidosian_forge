from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
def _get_submit_location(self, branch):
    submit_location = branch.get_submit_branch()
    location_type = 'submit branch'
    if submit_location is None:
        submit_location = branch.get_parent()
        location_type = 'parent branch'
    if submit_location is None:
        raise errors.NoSubmitBranch(branch)
    trace.note(gettext('Using {0} {1}').format(location_type, submit_location))
    return submit_location