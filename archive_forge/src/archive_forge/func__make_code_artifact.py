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
def _make_code_artifact(api: Api, job_builder: JobBuilder, run: 'wandb.sdk.wandb_run.Run', path: str, entrypoint: Optional[str], entity: Optional[str], project: Optional[str], name: Optional[str]) -> Optional[str]:
    """Helper for creating and logging code artifacts.

    Returns the name of the eventual job.
    """
    artifact_name = _make_code_artifact_name(os.path.join(path, entrypoint or ''), name)
    code_artifact = wandb.Artifact(name=artifact_name, type='code', description='Code artifact for job')
    path, entrypoint = _handle_artifact_entrypoint(path, entrypoint)
    try:
        code_artifact.add_dir(path)
    except Exception as e:
        if os.path.islink(path):
            wandb.termerror('Symlinks are not supported for code artifact jobs, please copy the code into a directory and try again')
        wandb.termerror(f'Error adding to code artifact: {e}')
        return None
    res, _ = api.create_artifact(artifact_type_name='code', artifact_collection_name=artifact_name, digest=code_artifact.digest, client_id=code_artifact._client_id, sequence_client_id=code_artifact._sequence_client_id, entity_name=entity, project_name=project, run_name=run.id, description='Code artifact for job', metadata={'codePath': path, 'entrypoint': entrypoint}, is_user_created=True, aliases=[{'artifactCollectionName': artifact_name, 'alias': a} for a in ['latest']])
    run.log_artifact(code_artifact)
    code_artifact.wait()
    job_builder._handle_server_artifact(res, code_artifact)
    if not name:
        name = code_artifact.name.replace('code', 'job').split(':')[0]
    return name