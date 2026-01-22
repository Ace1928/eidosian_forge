from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
class RevisionSpec_date(RevisionSpec):
    """Selects a revision on the basis of a datestamp."""
    help_txt = "Selects a revision on the basis of a datestamp.\n\n    Supply a datestamp to select the first revision that matches the date.\n    Date can be 'yesterday', 'today', 'tomorrow' or a YYYY-MM-DD string.\n    Matches the first entry after a given date (either at midnight or\n    at a specified time).\n\n    One way to display all the changes since yesterday would be::\n\n        brz log -r date:yesterday..\n\n    Examples::\n\n      date:yesterday            -> select the first revision since yesterday\n      date:2006-08-14,17:10:14  -> select the first revision after\n                                   August 14th, 2006 at 5:10pm.\n    "
    prefix = 'date:'

    def _scan_backwards(self, branch, dt):
        with branch.lock_read():
            graph = branch.repository.get_graph()
            last_match = None
            for revid in graph.iter_lefthand_ancestry(branch.last_revision(), (_mod_revision.NULL_REVISION,)):
                r = branch.repository.get_revision(revid)
                if r.datetime() < dt:
                    if last_match is None:
                        raise InvalidRevisionSpec(self.user_spec, branch)
                    return RevisionInfo(branch, None, last_match)
                last_match = revid
            return RevisionInfo(branch, None, last_match)

    def _bisect_backwards(self, branch, dt, hi):
        import bisect
        with branch.lock_read():
            rev = bisect.bisect(_RevListToTimestamps(branch), dt, 1, hi)
        if rev == branch.revno():
            raise InvalidRevisionSpec(self.user_spec, branch)
        return RevisionInfo(branch, rev)

    def _match_on(self, branch, revs):
        """Spec for date revisions:
          date:value
          value can be 'yesterday', 'today', 'tomorrow' or a YYYY-MM-DD string.
          matches the first entry after a given date (either at midnight or
          at a specified time).
        """
        try:
            dt = _parse_datespec(self.spec)
        except ValueError:
            raise InvalidRevisionSpec(self.user_spec, branch, 'invalid date')
        revno = branch.revno()
        if revno is None:
            return self._scan_backwards(branch, dt)
        else:
            return self._bisect_backwards(branch, dt, revno)