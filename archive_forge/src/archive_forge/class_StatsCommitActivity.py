from datetime import datetime
from typing import Any, Dict
import github.GithubObject
from github.GithubObject import Attribute
class StatsCommitActivity(github.GithubObject.NonCompletableGithubObject):
    """
    This class represents StatsCommitActivities. The reference can be found here https://docs.github.com/en/rest/reference/repos#get-the-last-year-of-commit-activity
    """

    def _initAttributes(self) -> None:
        self._week: Attribute[datetime] = github.GithubObject.NotSet
        self._total: Attribute[int] = github.GithubObject.NotSet
        self._days: Attribute[int] = github.GithubObject.NotSet

    @property
    def week(self) -> datetime:
        return self._week.value

    @property
    def total(self) -> int:
        return self._total.value

    @property
    def days(self) -> int:
        return self._days.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if 'week' in attributes:
            self._week = self._makeTimestampAttribute(attributes['week'])
        if 'total' in attributes:
            self._total = self._makeIntAttribute(attributes['total'])
        if 'days' in attributes:
            self._days = self._makeListOfIntsAttribute(attributes['days'])