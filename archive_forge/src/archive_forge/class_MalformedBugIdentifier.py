from . import errors, registry, urlutils
class MalformedBugIdentifier(errors.BzrError):
    _fmt = 'Did not understand bug identifier %(bug_id)s: %(reason)s. See "brz help bugs" for more information on this feature.'

    def __init__(self, bug_id, reason):
        self.bug_id = bug_id
        self.reason = reason