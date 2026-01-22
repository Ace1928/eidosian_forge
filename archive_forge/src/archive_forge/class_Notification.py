from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.Issue
import github.NotificationSubject
import github.PullRequest
import github.Repository
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
class Notification(CompletableGithubObject):
    """
    This class represents Notifications. The reference can be found here https://docs.github.com/en/rest/reference/activity#notifications
    """

    def _initAttributes(self) -> None:
        self._id: Attribute[str] = NotSet
        self._last_read_at: Attribute[datetime] = NotSet
        self._repository: Attribute[github.Repository.Repository] = NotSet
        self._subject: Attribute[github.NotificationSubject.NotificationSubject] = NotSet
        self._reason: Attribute[str] = NotSet
        self._subscription_url: Attribute[str] = NotSet
        self._unread: Attribute[bool] = NotSet
        self._updated_at: Attribute[datetime] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'id': self._id.value, 'subject': self._subject.value})

    @property
    def id(self) -> str:
        self._completeIfNotSet(self._id)
        return self._id.value

    @property
    def last_read_at(self) -> datetime:
        self._completeIfNotSet(self._last_read_at)
        return self._last_read_at.value

    @property
    def repository(self) -> github.Repository.Repository:
        self._completeIfNotSet(self._repository)
        return self._repository.value

    @property
    def subject(self) -> github.NotificationSubject.NotificationSubject:
        self._completeIfNotSet(self._subject)
        return self._subject.value

    @property
    def reason(self) -> str:
        self._completeIfNotSet(self._reason)
        return self._reason.value

    @property
    def subscription_url(self) -> str:
        self._completeIfNotSet(self._subscription_url)
        return self._subscription_url.value

    @property
    def unread(self) -> bool:
        self._completeIfNotSet(self._unread)
        return self._unread.value

    @property
    def updated_at(self) -> datetime:
        self._completeIfNotSet(self._updated_at)
        return self._updated_at.value

    @property
    def url(self) -> str:
        self._completeIfNotSet(self._url)
        return self._url.value

    def mark_as_read(self) -> None:
        """
        :calls: `PATCH /notifications/threads/{id} <https://docs.github.com/en/rest/reference/activity#notifications>`_
        """
        headers, data = self._requester.requestJsonAndCheck('PATCH', self.url)

    def get_pull_request(self) -> github.PullRequest.PullRequest:
        headers, data = self._requester.requestJsonAndCheck('GET', self.subject.url)
        return github.PullRequest.PullRequest(self._requester, headers, data, completed=True)

    def get_issue(self) -> github.Issue.Issue:
        headers, data = self._requester.requestJsonAndCheck('GET', self.subject.url)
        return github.Issue.Issue(self._requester, headers, data, completed=True)

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'id' in attributes:
            self._id = self._makeStringAttribute(attributes['id'])
        if 'last_read_at' in attributes:
            self._last_read_at = self._makeDatetimeAttribute(attributes['last_read_at'])
        if 'repository' in attributes:
            self._repository = self._makeClassAttribute(github.Repository.Repository, attributes['repository'])
        if 'subject' in attributes:
            self._subject = self._makeClassAttribute(github.NotificationSubject.NotificationSubject, attributes['subject'])
        if 'reason' in attributes:
            self._reason = self._makeStringAttribute(attributes['reason'])
        if 'subscription_url' in attributes:
            self._subscription_url = self._makeStringAttribute(attributes['subscription_url'])
        if 'unread' in attributes:
            self._unread = self._makeBoolAttribute(attributes['unread'])
        if 'updated_at' in attributes:
            self._updated_at = self._makeDatetimeAttribute(attributes['updated_at'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])