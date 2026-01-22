from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable
import github.AdvisoryVulnerability
import github.NamedUser
from github.AdvisoryBase import AdvisoryBase
from github.AdvisoryCredit import AdvisoryCredit, Credit
from github.AdvisoryCreditDetailed import AdvisoryCreditDetailed
from github.GithubObject import Attribute, NotSet, Opt
def add_vulnerabilities(self, vulnerabilities: Iterable[AdvisoryVulnerabilityInput]) -> None:
    """
        :calls: `PATCH /repos/{owner}/{repo}/security-advisories/:advisory_id <https://docs.github.com/en/rest/security-advisories/repository-advisories>`
        """
    assert isinstance(vulnerabilities, Iterable), vulnerabilities
    for vulnerability in vulnerabilities:
        github.AdvisoryVulnerability.AdvisoryVulnerability._validate_vulnerability(vulnerability)
    post_parameters = {'vulnerabilities': [github.AdvisoryVulnerability.AdvisoryVulnerability._to_github_dict(vulnerability) for vulnerability in self.vulnerabilities + list(vulnerabilities)]}
    headers, data = self._requester.requestJsonAndCheck('PATCH', self.url, input=post_parameters)
    self._useAttributes(data)