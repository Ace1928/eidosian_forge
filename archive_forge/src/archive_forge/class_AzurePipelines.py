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
class AzurePipelines(CIProvider):
    """CI provider implementation for Azure Pipelines."""

    def __init__(self) -> None:
        self.auth = AzurePipelinesAuthHelper()
        self._changes: AzurePipelinesChanges | None = None

    @staticmethod
    def is_supported() -> bool:
        """Return True if this provider is supported in the current running environment."""
        return os.environ.get('SYSTEM_COLLECTIONURI', '').startswith('https://dev.azure.com/')

    @property
    def code(self) -> str:
        """Return a unique code representing this provider."""
        return CODE

    @property
    def name(self) -> str:
        """Return descriptive name for this provider."""
        return 'Azure Pipelines'

    def generate_resource_prefix(self) -> str:
        """Return a resource prefix specific to this CI provider."""
        try:
            prefix = 'azp-%s-%s-%s' % (os.environ['BUILD_BUILDID'], os.environ['SYSTEM_JOBATTEMPT'], os.environ['SYSTEM_JOBIDENTIFIER'])
        except KeyError as ex:
            raise MissingEnvironmentVariable(name=ex.args[0]) from None
        return prefix

    def get_base_commit(self, args: CommonConfig) -> str:
        """Return the base commit or an empty string."""
        return self._get_changes(args).base_commit or ''

    def _get_changes(self, args: CommonConfig) -> AzurePipelinesChanges:
        """Return an AzurePipelinesChanges instance, which will be created on first use."""
        if not self._changes:
            self._changes = AzurePipelinesChanges(args)
        return self._changes

    def detect_changes(self, args: TestConfig) -> t.Optional[list[str]]:
        """Initialize change detection."""
        result = self._get_changes(args)
        if result.is_pr:
            job_type = 'pull request'
        else:
            job_type = 'merge commit'
        display.info('Processing %s for branch %s commit %s' % (job_type, result.branch, result.commit))
        if not args.metadata.changes:
            args.metadata.populate_changes(result.diff)
        if result.paths is None:
            display.warning('No successful commit found. All tests will be executed.')
        return result.paths

    def supports_core_ci_auth(self) -> bool:
        """Return True if Ansible Core CI is supported."""
        return True

    def prepare_core_ci_auth(self) -> dict[str, t.Any]:
        """Return authentication details for Ansible Core CI."""
        try:
            request = dict(org_name=os.environ['SYSTEM_COLLECTIONURI'].strip('/').split('/')[-1], project_name=os.environ['SYSTEM_TEAMPROJECT'], build_id=int(os.environ['BUILD_BUILDID']), task_id=str(uuid.UUID(os.environ['SYSTEM_TASKINSTANCEID'])))
        except KeyError as ex:
            raise MissingEnvironmentVariable(name=ex.args[0]) from None
        self.auth.sign_request(request)
        auth = dict(azp=request)
        return auth

    def get_git_details(self, args: CommonConfig) -> t.Optional[dict[str, t.Any]]:
        """Return details about git in the current environment."""
        changes = self._get_changes(args)
        details = dict(base_commit=changes.base_commit, commit=changes.commit)
        return details