from __future__ import annotations
from datetime import datetime
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class StatsCodeFrequency(NonCompletableGithubObject):
    """
    This class represents statistics of StatsCodeFrequencies.
    The reference can be found here https://docs.github.com/en/rest/metrics/statistics?apiVersion=2022-11-28#get-the-weekly-commit-activity
    """

    def _initAttributes(self) -> None:
        self._week: Attribute[datetime] = NotSet
        self._additions: Attribute[int] = NotSet
        self._deletions: Attribute[int] = NotSet

    @property
    def week(self) -> datetime:
        return self._week.value

    @property
    def additions(self) -> int:
        return self._additions.value

    @property
    def deletions(self) -> int:
        return self._deletions.value

    def _useAttributes(self, attributes: tuple[int, int, int]) -> None:
        self._week = self._makeTimestampAttribute(attributes[0])
        self._additions = self._makeIntAttribute(attributes[1])
        self._deletions = self._makeIntAttribute(attributes[2])