from __future__ import annotations
import urllib.parse
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.Event
import github.Gist
import github.GithubObject
import github.Organization
import github.PaginatedList
import github.Permissions
import github.Plan
import github.Repository
from github import Consts
from github.GithubObject import Attribute, NotSet, Opt, is_defined
from github.PaginatedList import PaginatedList

        :calls: `GET /orgs/{org}/memberships/{username} <https://docs.github.com/en/rest/reference/orgs#check-organization-membership-for-a-user>`_
        