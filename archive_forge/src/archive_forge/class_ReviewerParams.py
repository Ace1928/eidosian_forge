from __future__ import annotations
from typing import Any
import github.NamedUser
import github.Team
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class ReviewerParams:
    """
    This class presents reviewers as can be configured for an Environment.
    """

    def __init__(self, type_: str, id_: int):
        assert isinstance(type_, str) and type_ in ('User', 'Team')
        assert isinstance(id_, int)
        self.type = type_
        self.id = id_

    def _asdict(self) -> dict:
        return {'type': self.type, 'id': self.id}