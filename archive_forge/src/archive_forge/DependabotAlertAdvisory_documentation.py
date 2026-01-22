from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.AdvisoryBase
import github.DependabotAlertVulnerability
from github.GithubObject import Attribute, NotSet

    This class represents a package flagged by a Dependabot alert that is vulnerable to a parent SecurityAdvisory.
    The reference can be found here https://docs.github.com/en/rest/dependabot/alerts
    