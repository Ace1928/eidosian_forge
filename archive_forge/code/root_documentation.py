from .base import Submodule, UpdateProgress
from .util import find_first_remote_branch
from git.exc import InvalidGitRepositoryError
import git
import logging
from typing import TYPE_CHECKING, Union
from git.types import Commit_ish
:return: The actual repository containing the submodules