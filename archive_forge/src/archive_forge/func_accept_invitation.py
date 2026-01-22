from __future__ import annotations
import urllib.parse
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, NamedTuple
import github.Authorization
import github.Event
import github.Gist
import github.GithubObject
import github.Invitation
import github.Issue
import github.Membership
import github.Migration
import github.NamedUser
import github.Notification
import github.Organization
import github.Plan
import github.Repository
import github.UserKey
from github import Consts
from github.GithubObject import (
from github.PaginatedList import PaginatedList
def accept_invitation(self, invitation: Invitation | int) -> None:
    """
        :calls: `PATCH /user/repository_invitations/{invitation_id} <https://docs.github.com/en/rest/reference/repos/invitations#>`_
        """
    assert isinstance(invitation, github.Invitation.Invitation) or isinstance(invitation, int)
    if isinstance(invitation, github.Invitation.Invitation):
        invitation = invitation.id
    headers, data = self._requester.requestJsonAndCheck('PATCH', f'/user/repository_invitations/{invitation}', input={})