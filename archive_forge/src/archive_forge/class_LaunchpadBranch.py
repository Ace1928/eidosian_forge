import base64
import re
from urllib.parse import urlparse, urlunparse
from ... import bedding, branch, errors, osutils, trace, transport
from ...i18n import gettext
from launchpadlib.credentials import (AccessToken, Credentials,
from launchpadlib.launchpad import Launchpad
class LaunchpadBranch:
    """Provide bzr and lp API access to a Launchpad branch."""

    def __init__(self, lp_branch, bzr_url, bzr_branch=None, check_update=True):
        """Constructor.

        :param lp_branch: The Launchpad branch.
        :param bzr_url: The URL of the Bazaar branch.
        :param bzr_branch: An instance of the Bazaar branch.
        """
        self.bzr_url = bzr_url
        self._bzr = bzr_branch
        self._push_bzr = None
        self._check_update = check_update
        self.lp = lp_branch

    @property
    def bzr(self):
        """Return the bzr branch for this branch."""
        if self._bzr is None:
            self._bzr = branch.Branch.open(self.bzr_url)
        return self._bzr

    @property
    def push_bzr(self):
        """Return the push branch for this branch."""
        if self._push_bzr is None:
            self._push_bzr = branch.Branch.open(self.lp.bzr_identity)
        return self._push_bzr

    @staticmethod
    def plausible_launchpad_url(url):
        """Is 'url' something that could conceivably be pushed to LP?

        :param url: A URL that may refer to a Launchpad branch.
        :return: A boolean.
        """
        if url is None:
            return False
        if url.startswith('lp:'):
            return True
        regex = re.compile('([a-z]*\\+)*(bzr\\+ssh|http)://bazaar.*.launchpad.net')
        return bool(regex.match(url))

    def get_target(self):
        """Return the 'LaunchpadBranch' for the target of this one."""
        lp_branch = self.lp
        if lp_branch.project is not None:
            dev_focus = lp_branch.project.development_focus
            if dev_focus is None:
                raise errors.BzrError(gettext('%s has no development focus.') % lp_branch.bzr_identity)
            target = dev_focus.branch
            if target is None:
                raise errors.BzrError(gettext('development focus %s has no branch.') % dev_focus)
        elif lp_branch.sourcepackage is not None:
            target = lp_branch.sourcepackage.getBranch(pocket='Release')
            if target is None:
                raise errors.BzrError(gettext('source package %s has no branch.') % lp_branch.sourcepackage)
        else:
            raise errors.BzrError(gettext('%s has no associated product or source package.') % lp_branch.bzr_identity)
        return LaunchpadBranch(target, target.bzr_identity)

    def update_lp(self):
        """Update the Launchpad copy of this branch."""
        if not self._check_update:
            return
        with self.bzr.lock_read():
            if self.lp.last_scanned_id is not None:
                if self.bzr.last_revision() == self.lp.last_scanned_id:
                    trace.note(gettext('%s is already up-to-date.') % self.lp.bzr_identity)
                    return
                graph = self.bzr.repository.get_graph()
                if not graph.is_ancestor(osutils.safe_utf8(self.lp.last_scanned_id), self.bzr.last_revision()):
                    raise errors.DivergedBranches(self.bzr, self.push_bzr)
                trace.note(gettext('Pushing to %s') % self.lp.bzr_identity)
            self.bzr.push(self.push_bzr)

    def find_lca_tree(self, other):
        """Find the revision tree for the LCA of this branch and other.

        :param other: Another LaunchpadBranch
        :return: The RevisionTree of the LCA of this branch and other.
        """
        graph = self.bzr.repository.get_graph(other.bzr.repository)
        lca = graph.find_unique_lca(self.bzr.last_revision(), other.bzr.last_revision())
        return self.bzr.repository.revision_tree(lca)