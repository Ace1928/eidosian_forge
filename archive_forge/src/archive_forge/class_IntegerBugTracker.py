from . import errors, registry, urlutils
class IntegerBugTracker(BugTracker):
    """A bug tracker that only allows integer bug IDs."""

    def check_bug_id(self, bug_id):
        try:
            int(bug_id)
        except ValueError as exc:
            raise MalformedBugIdentifier(bug_id, 'Must be an integer') from exc