import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Optional, Set, Tuple, Union
from ._commit_api import CommitOperationAdd, CommitOperationDelete
from .community import DiscussionWithDetails
from .utils import experimental
from .utils._cache_manager import _format_size
from .utils.insecure_hashlib import sha256
@dataclass
class MultiCommitStrategy:
    """Dataclass containing a list of [`MultiCommitStep`] to commit iteratively.

    A strategy is identified by its own deterministic ID based on the list of its steps (hexadecimal sha256). ID is
    persistent between re-runs if the list of commits is kept the same.
    """
    addition_commits: List[MultiCommitStep]
    deletion_commits: List[MultiCommitStep]
    id: str = field(init=False)
    all_steps: Set[str] = field(init=False)

    def __post_init__(self) -> None:
        self.all_steps = {step.id for step in self.addition_commits + self.deletion_commits}
        if len(self.all_steps) < len(self.addition_commits) + len(self.deletion_commits):
            raise ValueError('Got duplicate commits in MultiCommitStrategy. All commits must be unique.')
        if len(self.all_steps) == 0:
            raise ValueError('A MultiCommitStrategy must have at least 1 commit, got 0.')
        sha = sha256()
        for step in self.addition_commits + self.deletion_commits:
            sha.update('new step'.encode())
            sha.update(step.id.encode())
        self.id = sha.hexdigest()