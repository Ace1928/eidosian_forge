from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.WorkflowRun
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def workflow_run(self) -> WorkflowRun:
    return self._workflow_run.value