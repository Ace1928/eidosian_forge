from __future__ import annotations
import urllib.parse
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.Event
import github.GithubObject
import github.HookDelivery
import github.NamedUser
import github.OrganizationDependabotAlert
import github.OrganizationSecret
import github.OrganizationVariable
import github.Plan
import github.Project
import github.Repository
import github.Team
from github import Consts
from github.GithubObject import (
from github.PaginatedList import PaginatedList

        :calls: `GET /orgs/{org}/dependabot/alerts <https://docs.github.com/en/rest/dependabot/alerts#list-dependabot-alerts-for-an-organization>`_
        :param state: Optional string
        :param severity: Optional string
        :param ecosystem: Optional string
        :param package: Optional string
        :param scope: Optional string
        :param sort: Optional string
        :param direction: Optional string
        :rtype: :class:`PaginatedList` of :class:`github.DependabotAlert.DependabotAlert`
        