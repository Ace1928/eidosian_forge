from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.AdvisoryVulnerabilityPackage
import github.DependabotAlertAdvisory
import github.DependabotAlertDependency
import github.DependabotAlertVulnerability
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

    This class represents a DependabotAlert.
    The reference can be found here https://docs.github.com/en/rest/dependabot/alerts
    