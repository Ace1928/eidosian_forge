import contextlib
import os
from dulwich.refs import SymrefLoop
from .. import branch as _mod_branch
from .. import errors as brz_errors
from .. import osutils, trace, urlutils
from ..controldir import (BranchReferenceLoop, ControlDir, ControlDirFormat,
from ..transport import (FileExists, NoSuchFile, do_catching_redirections,
from .mapping import decode_git_path, encode_git_path
from .push import GitPushResult
from .transportgit import OBJECTDIR, TransportObjectStore
class UseExistingRepository(RepositoryAcquisitionPolicy):
    """A policy of reusing an existing repository"""

    def __init__(self, repository, stack_on=None, stack_on_pwd=None, require_stacking=False):
        """Constructor.

        :param repository: The repository to use.
        :param stack_on: A location to stack on
        :param stack_on_pwd: If stack_on is relative, the location it is
            relative to.
        """
        super().__init__(stack_on, stack_on_pwd, require_stacking)
        self._repository = repository

    def acquire_repository(self, make_working_trees=None, shared=False, possible_transports=None):
        """Implementation of RepositoryAcquisitionPolicy.acquire_repository

        Returns an existing repository to use.
        """
        return (self._repository, False)