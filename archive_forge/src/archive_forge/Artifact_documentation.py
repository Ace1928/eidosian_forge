from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.WorkflowRun
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

        :calls: `DELETE /repos/{owner}/{repo}/actions/artifacts/{artifact_id} <https://docs.github.com/en/rest/actions/artifacts#delete-an-artifact>`_
        