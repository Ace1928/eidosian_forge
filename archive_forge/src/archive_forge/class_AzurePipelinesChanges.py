from __future__ import annotations
import os
import tempfile
import uuid
import typing as t
import urllib.parse
from ..encoding import (
from ..config import (
from ..git import (
from ..http import (
from ..util import (
from . import (
class AzurePipelinesChanges:
    """Change information for an Azure Pipelines build."""

    def __init__(self, args: CommonConfig) -> None:
        self.args = args
        self.git = Git()
        try:
            self.org_uri = os.environ['SYSTEM_COLLECTIONURI']
            self.project = os.environ['SYSTEM_TEAMPROJECT']
            self.repo_type = os.environ['BUILD_REPOSITORY_PROVIDER']
            self.source_branch = os.environ['BUILD_SOURCEBRANCH']
            self.source_branch_name = os.environ['BUILD_SOURCEBRANCHNAME']
            self.pr_branch_name = os.environ.get('SYSTEM_PULLREQUEST_TARGETBRANCH')
        except KeyError as ex:
            raise MissingEnvironmentVariable(name=ex.args[0]) from None
        if self.source_branch.startswith('refs/tags/'):
            raise ChangeDetectionNotSupported('Change detection is not supported for tags.')
        self.org = self.org_uri.strip('/').split('/')[-1]
        self.is_pr = self.pr_branch_name is not None
        if self.is_pr:
            self.branch = self.pr_branch_name
            self.base_commit = 'HEAD^1'
            self.commit = 'HEAD^2'
        else:
            commits = self.get_successful_merge_run_commits()
            self.branch = self.source_branch_name
            self.base_commit = self.get_last_successful_commit(commits)
            self.commit = 'HEAD'
        self.commit = self.git.run_git(['rev-parse', self.commit]).strip()
        if self.base_commit:
            self.base_commit = self.git.run_git(['rev-parse', self.base_commit]).strip()
            dot_range = '%s...%s' % (self.base_commit, self.commit)
            self.paths = sorted(self.git.get_diff_names([dot_range]))
            self.diff = self.git.get_diff([dot_range])
        else:
            self.paths = None
            self.diff = []

    def get_successful_merge_run_commits(self) -> set[str]:
        """Return a set of recent successsful merge commits from Azure Pipelines."""
        parameters = dict(maxBuildsPerDefinition=100, queryOrder='queueTimeDescending', resultFilter='succeeded', reasonFilter='batchedCI', repositoryType=self.repo_type, repositoryId='%s/%s' % (self.org, self.project))
        url = '%s%s/_apis/build/builds?api-version=6.0&%s' % (self.org_uri, self.project, urllib.parse.urlencode(parameters))
        http = HttpClient(self.args, always=True)
        response = http.get(url)
        try:
            result = response.json()
        except Exception:
            display.warning('Unable to find project. Cannot determine changes. All tests will be executed.')
            return set()
        commits = set((build['sourceVersion'] for build in result['value']))
        return commits

    def get_last_successful_commit(self, commits: set[str]) -> t.Optional[str]:
        """Return the last successful commit from git history that is found in the given commit list, or None."""
        commit_history = self.git.get_rev_list(max_count=100)
        ordered_successful_commits = [commit for commit in commit_history if commit in commits]
        last_successful_commit = ordered_successful_commits[0] if ordered_successful_commits else None
        return last_successful_commit