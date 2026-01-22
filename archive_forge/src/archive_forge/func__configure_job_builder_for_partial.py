import json
import logging
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import wandb
from wandb.apis.internal import Api
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.internal.job_builder import JobBuilder
from wandb.sdk.launch.builder.build import get_current_python_version
from wandb.sdk.launch.git_reference import GitReference
from wandb.sdk.launch.utils import _is_git_uri
from wandb.sdk.lib import filesystem
from wandb.util import make_artifact_name_safe
def _configure_job_builder_for_partial(tmpdir: str, job_source: str) -> JobBuilder:
    """Configure job builder with temp dir and job source."""
    if job_source == 'git':
        job_source = 'repo'
    if job_source == 'code':
        job_source = 'artifact'
    settings = wandb.Settings()
    settings.update({'files_dir': tmpdir, 'job_source': job_source})
    job_builder = JobBuilder(settings=settings)
    job_builder._is_notebook_run = False
    job_builder.set_config({})
    job_builder.set_summary({})
    return job_builder