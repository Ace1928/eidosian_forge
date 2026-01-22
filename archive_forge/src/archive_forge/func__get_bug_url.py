from . import errors, registry, urlutils
def _get_bug_url(self, bug_id):
    """Given a validated bug_id, return the bug's web page's URL."""
    if '{id}' not in self._base_url:
        raise InvalidBugTrackerURL(self._abbreviation, self._base_url)
    return self._base_url.replace('{id}', str(bug_id))