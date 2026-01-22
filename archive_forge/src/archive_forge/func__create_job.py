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
def _create_job(api: Api, job_type: str, path: str, entity: Optional[str]=None, project: Optional[str]=None, name: Optional[str]=None, description: Optional[str]=None, aliases: Optional[List[str]]=None, runtime: Optional[str]=None, entrypoint: Optional[str]=None, git_hash: Optional[str]=None) -> Tuple[Optional[Artifact], str, List[str]]:
    wandb.termlog(f'Creating launch job of type: {job_type}...')
    if name and name != make_artifact_name_safe(name):
        wandb.termerror(f'Artifact names may only contain alphanumeric characters, dashes, underscores, and dots. Did you mean: {make_artifact_name_safe(name)}')
        return (None, '', [])
    aliases = aliases or []
    tempdir = tempfile.TemporaryDirectory()
    try:
        metadata, requirements = _make_metadata_for_partial_job(job_type=job_type, tempdir=tempdir, git_hash=git_hash, runtime=runtime, path=path, entrypoint=entrypoint)
        if not metadata:
            return (None, '', [])
    except Exception as e:
        wandb.termerror(f'Error creating job: {e}')
        return (None, '', [])
    _dump_metadata_and_requirements(metadata=metadata, tmp_path=tempdir.name, requirements=requirements)
    try:
        run = wandb.init(dir=tempdir.name, settings={'silent': True, 'disable_job_creation': True}, entity=entity, project=project, job_type='cli_create_job')
    except Exception:
        return (None, '', [])
    job_builder = _configure_job_builder_for_partial(tempdir.name, job_source=job_type)
    if job_type == 'code':
        job_name = _make_code_artifact(api=api, job_builder=job_builder, path=path, entrypoint=entrypoint, run=run, entity=entity, project=project, name=name)
        if not job_name:
            return (None, '', [])
        name = job_name
    artifact = job_builder.build()
    if not artifact:
        wandb.termerror('JobBuilder failed to build a job')
        _logger.debug('Failed to build job, check job source and metadata')
        return (None, '', [])
    if not name:
        name = artifact.name
    aliases += job_builder._aliases
    if 'latest' not in aliases:
        aliases += ['latest']
    res, _ = api.create_artifact(artifact_type_name='job', artifact_collection_name=name, digest=artifact.digest, client_id=artifact._client_id, sequence_client_id=artifact._sequence_client_id, entity_name=entity, project_name=project, run_name=run.id, description=description, metadata=metadata, is_user_created=True, aliases=[{'artifactCollectionName': name, 'alias': a} for a in aliases])
    action = 'No changes detected for'
    if not res.get('artifactSequence', {}).get('latestArtifact'):
        action = 'Created'
    elif res.get('state') == 'PENDING':
        action = 'Updated'
    run.log_artifact(artifact, aliases=aliases)
    artifact.wait()
    run.finish()
    _run = wandb.Api().run(f'{entity}/{project}/{run.id}')
    _run.delete()
    return (artifact, action, aliases)