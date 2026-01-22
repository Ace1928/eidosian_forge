from . import errors, registry, urlutils
class URLParametrizedBugTracker(BugTracker):
    """A type of bug tracker that can be found on a variety of different sites,
    and thus needs to have the base URL configured.

    Looks for a config setting in the form '<type_name>_<abbreviation>_url'.
    `type_name` is the name of the type of tracker and `abbreviation`
    is a short name for the particular instance.
    """

    def get(self, abbreviation, branch):
        config = branch.get_config()
        url = config.get_user_option('{}_{}_url'.format(self.type_name, abbreviation), expand=False)
        if url is None:
            return None
        self._base_url = url
        return self

    def __init__(self, type_name, bug_area):
        self.type_name = type_name
        self._bug_area = bug_area

    def _get_bug_url(self, bug_id):
        """Return a URL for a bug on this Trac instance."""
        return urlutils.join(self._base_url, self._bug_area) + str(bug_id)