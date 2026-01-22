from dulwich import errors as git_errors
from .. import errors as brz_errors
class BzrGitError(brz_errors.BzrError):
    """The base-level exception for bzr-git errors."""