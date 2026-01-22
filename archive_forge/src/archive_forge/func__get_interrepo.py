from ...controldir import format_registry
from ...repository import InterRepository
from ...tests import TestCaseWithTransport
from ..interrepo import InterToGitRepository
from ..mapping import BzrGitMappingExperimental, BzrGitMappingv1
def _get_interrepo(self, mapping=None):
    self.bzr_repo.lock_read()
    self.addCleanup(self.bzr_repo.unlock)
    interrepo = InterRepository.get(self.bzr_repo, self.git_repo)
    if mapping is not None:
        interrepo.mapping = mapping
    return interrepo