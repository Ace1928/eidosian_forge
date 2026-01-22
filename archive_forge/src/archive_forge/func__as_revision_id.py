from ..errors import InvalidRevisionId
from ..revision import NULL_REVISION
from ..revisionspec import InvalidRevisionSpec, RevisionInfo, RevisionSpec
def _as_revision_id(self, context_branch):
    loc = self.spec.find(':')
    git_sha1 = self.spec[loc + 1:].encode('utf-8')
    if len(git_sha1) > 40 or len(git_sha1) < 4 or (not valid_git_sha1(git_sha1)):
        raise InvalidRevisionSpec(self.user_spec, context_branch)
    from . import lazy_check_versions
    lazy_check_versions()
    if len(git_sha1) == 40:
        return self._lookup_git_sha1(context_branch, git_sha1)
    else:
        return self._find_short_git_sha1(context_branch, git_sha1)