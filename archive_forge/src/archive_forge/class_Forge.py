import re
from typing import Optional, Type
from . import errors, hooks, registry, urlutils
class Forge:
    """A hosting site manager.
    """
    supports_merge_proposal_labels: bool
    supports_merge_proposal_title: bool

    @property
    def name(self):
        """Name of this instance."""
        return '{} at {}'.format(type(self).__name__, self.base_url)
    supports_merge_proposal_commit_message: bool
    base_url: str
    merge_proposal_description_format: str
    supports_allow_collaboration: bool = False

    def publish_derived(self, new_branch, base_branch, name, project=None, owner=None, revision_id=None, overwrite=False, allow_lossy=True, tag_selector=None):
        """Publish a branch to the site, derived from base_branch.

        :param base_branch: branch to derive the new branch from
        :param new_branch: branch to publish
        :return: resulting branch, public URL
        :raise ForgeLoginRequired: Action requires a forge login, but none is
            known.
        """
        raise NotImplementedError(self.publish_derived)

    def get_derived_branch(self, base_branch, name, project=None, owner=None, preferred_schemes=None):
        """Get a derived branch ('a fork').
        """
        raise NotImplementedError(self.get_derived_branch)

    def get_push_url(self, branch):
        """Get the push URL for a branch."""
        raise NotImplementedError(self.get_push_url)

    def get_web_url(self, branch):
        """Get the web viewing URL for a branch."""
        raise NotImplementedError(self.get_web_url)

    def get_proposer(self, source_branch, target_branch):
        """Get a merge proposal creator.

        :note: source_branch does not have to be hosted by the forge.

        :param source_branch: Source branch
        :param target_branch: Target branch
        :return: A MergeProposalBuilder object
        """
        raise NotImplementedError(self.get_proposer)

    def iter_proposals(self, source_branch, target_branch, status='open'):
        """Get the merge proposals for a specified branch tuple.

        :param source_branch: Source branch
        :param target_branch: Target branch
        :param status: Status of proposals to iterate over
        :return: Iterate over MergeProposal object
        """
        raise NotImplementedError(self.iter_proposals)

    def get_proposal_by_url(self, url):
        """Retrieve a branch proposal by URL.

        :param url: Merge proposal URL.
        :return: MergeProposal object
        :raise UnsupportedForge: Forge does not support this URL
        """
        raise NotImplementedError(self.get_proposal_by_url)

    def hosts(self, branch):
        """Return true if this forge hosts given branch."""
        raise NotImplementedError(self.hosts)

    @classmethod
    def probe_from_hostname(cls, hostname, possible_transports=None):
        """Create a Forge object if this forge knows about a hostname.
        """
        raise NotImplementedError(cls.probe_from_hostname)

    @classmethod
    def probe_from_branch(cls, branch):
        """Create a Forge object if this forge knows about a branch."""
        url = urlutils.strip_segment_parameters(branch.user_url)
        return cls.probe_from_url(url, possible_transports=[branch.control_transport])

    @classmethod
    def probe_from_url(cls, url, possible_transports=None):
        """Create a Forge object if this forge knows about a URL."""
        hostname = urlutils.URL.from_string(url).host
        return cls.probe_from_hostname(hostname, possible_transports=possible_transports)

    def iter_my_proposals(self, status='open', author=None):
        """Iterate over the proposals created by the currently logged in user.

        :param status: Only yield proposals with this status
            (one of: 'open', 'closed', 'merged', 'all')
        :param author: Name of author to query (defaults to current user)
        :return: Iterator over MergeProposal objects
        :raise ForgeLoginRequired: Action requires a forge login, but none is
            known.
        """
        raise NotImplementedError(self.iter_my_proposals)

    def iter_my_forks(self, owner=None):
        """Iterate over the currently logged in users' forks.

        :param owner: Name of owner to query (defaults to current user)
        :return: Iterator over project_name
        """
        raise NotImplementedError(self.iter_my_forks)

    def delete_project(self, name):
        """Delete a project.
        """
        raise NotImplementedError(self.delete_project)

    def create_project(self, name, summary=None):
        """Create a project.
        """
        raise NotImplementedError(self.create_project)

    @classmethod
    def iter_instances(cls):
        """Iterate instances.

        :return: Forge instances
        """
        raise NotImplementedError(cls.iter_instances)

    def get_current_user(self):
        """Retrieve the name of the currently logged in user.

        :return: Username or None if not logged in
        """
        raise NotImplementedError(self.get_current_user)

    def get_user_url(self, user):
        """Rerieve the web URL for a user."""
        raise NotImplementedError(self.get_user_url)