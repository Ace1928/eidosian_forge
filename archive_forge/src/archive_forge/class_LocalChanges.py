from __future__ import annotations
import os
import platform
import random
import re
import typing as t
from ..config import (
from ..io import (
from ..git import (
from ..util import (
from . import (
class LocalChanges:
    """Change information for local work."""

    def __init__(self, args: TestConfig) -> None:
        self.args = args
        self.git = Git()
        self.current_branch = self.git.get_branch()
        if self.is_official_branch(self.current_branch):
            raise InvalidBranch(branch=self.current_branch, reason='Current branch is not a feature branch.')
        self.fork_branch = None
        self.fork_point = None
        self.local_branches = sorted(self.git.get_branches())
        self.official_branches = sorted([b for b in self.local_branches if self.is_official_branch(b)])
        for self.fork_branch in self.official_branches:
            try:
                self.fork_point = self.git.get_branch_fork_point(self.fork_branch)
                break
            except SubprocessError:
                pass
        if self.fork_point is None:
            raise ApplicationError('Unable to auto-detect fork branch and fork point.')
        self.tracked = sorted(self.git.get_file_names(['--cached']))
        self.untracked = sorted(self.git.get_file_names(['--others', '--exclude-standard']))
        self.committed = sorted(self.git.get_diff_names([self.fork_point, 'HEAD']))
        self.staged = sorted(self.git.get_diff_names(['--cached']))
        self.unstaged = sorted(self.git.get_diff_names([]))
        self.diff = self.git.get_diff([self.fork_point])

    def is_official_branch(self, name: str) -> bool:
        """Return True if the given branch name an official branch for development or releases."""
        if self.args.base_branch:
            return name == self.args.base_branch
        if name == 'devel':
            return True
        if re.match('^stable-[0-9]+\\.[0-9]+$', name):
            return True
        return False