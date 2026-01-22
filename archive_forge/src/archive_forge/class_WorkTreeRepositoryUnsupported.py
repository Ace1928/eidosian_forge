from gitdb.exc import (
from git.compat import safe_decode
from git.util import remove_password_if_present
from typing import List, Sequence, Tuple, Union, TYPE_CHECKING
from git.types import PathLike
class WorkTreeRepositoryUnsupported(InvalidGitRepositoryError):
    """Thrown to indicate we can't handle work tree repositories."""