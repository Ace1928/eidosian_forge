from __future__ import annotations
import collections
import urllib.parse
from base64 import b64encode
from collections.abc import Iterable
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any
from deprecated import deprecated
import github.AdvisoryCredit
import github.AdvisoryVulnerability
import github.Artifact
import github.AuthenticatedUser
import github.Autolink
import github.Branch
import github.CheckRun
import github.CheckSuite
import github.Clones
import github.CodeScanAlert
import github.Commit
import github.CommitComment
import github.Comparison
import github.ContentFile
import github.DependabotAlert
import github.Deployment
import github.Download
import github.Environment
import github.EnvironmentDeploymentBranchPolicy
import github.EnvironmentProtectionRule
import github.EnvironmentProtectionRuleReviewer
import github.Event
import github.GitBlob
import github.GitCommit
import github.GithubObject
import github.GitRef
import github.GitRelease
import github.GitReleaseAsset
import github.GitTag
import github.GitTree
import github.Hook
import github.HookDelivery
import github.Invitation
import github.Issue
import github.IssueComment
import github.IssueEvent
import github.Label
import github.License
import github.Milestone
import github.NamedUser
import github.Notification
import github.Organization
import github.PaginatedList
import github.Path
import github.Permissions
import github.Project
import github.PublicKey
import github.PullRequest
import github.PullRequestComment
import github.Referrer
import github.RepositoryAdvisory
import github.RepositoryKey
import github.RepositoryPreferences
import github.Secret
import github.SelfHostedActionsRunner
import github.SourceImport
import github.Stargazer
import github.StatsCodeFrequency
import github.StatsCommitActivity
import github.StatsContributor
import github.StatsParticipation
import github.StatsPunchCard
import github.Tag
import github.Team
import github.Variable
import github.View
import github.Workflow
import github.WorkflowRun
from github import Consts
from github.Environment import Environment
from github.GithubObject import (
from github.PaginatedList import PaginatedList
def __create_repository_advisory(self, summary: str, description: str, severity_or_cvss_vector_string: str, cve_id: str | None, vulnerabilities: Iterable[github.AdvisoryVulnerability.AdvisoryVulnerabilityInput] | None, cwe_ids: Iterable[str] | None, credits: Iterable[github.AdvisoryCredit.AdvisoryCredit] | None, private_vulnerability_reporting: bool) -> github.RepositoryAdvisory.RepositoryAdvisory:
    if vulnerabilities is None:
        vulnerabilities = []
    if cwe_ids is None:
        cwe_ids = []
    assert isinstance(summary, str), summary
    assert isinstance(description, str), description
    assert isinstance(severity_or_cvss_vector_string, str), severity_or_cvss_vector_string
    assert isinstance(cve_id, (str, type(None))), cve_id
    assert isinstance(vulnerabilities, Iterable), vulnerabilities
    for vulnerability in vulnerabilities:
        github.AdvisoryVulnerability.AdvisoryVulnerability._validate_vulnerability(vulnerability)
    assert isinstance(cwe_ids, Iterable), cwe_ids
    assert all((isinstance(element, str) for element in cwe_ids)), cwe_ids
    assert isinstance(credits, (Iterable, type(None))), credits
    if credits is not None:
        for credit in credits:
            github.AdvisoryCredit.AdvisoryCredit._validate_credit(credit)
    post_parameters = {'summary': summary, 'description': description, 'vulnerabilities': [github.AdvisoryVulnerability.AdvisoryVulnerability._to_github_dict(vulnerability) for vulnerability in vulnerabilities], 'cwe_ids': list(cwe_ids)}
    if cve_id is not None:
        post_parameters['cve_id'] = cve_id
    if credits is not None:
        post_parameters['credits'] = [github.AdvisoryCredit.AdvisoryCredit._to_github_dict(credit) for credit in credits]
    if severity_or_cvss_vector_string.startswith('CVSS:'):
        post_parameters['cvss_vector_string'] = severity_or_cvss_vector_string
    else:
        post_parameters['severity'] = severity_or_cvss_vector_string
    if private_vulnerability_reporting:
        headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/security-advisories/reports', input=post_parameters)
    else:
        headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/security-advisories', input=post_parameters)
    return github.RepositoryAdvisory.RepositoryAdvisory(self._requester, headers, data, completed=True)