import json
import logging
import os
import shutil
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Union
import yaml
import mlflow
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking._tracking_service.utils import _resolve_tracking_uri
from mlflow.tracking.artifact_utils import _download_artifact_from_uri, _upload_artifact_to_uri
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_runtime
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import TempDir
from mlflow.utils.uri import (
def update_model_requirements(model_uri: str, operation: Literal['add', 'remove'], requirement_list: List[str]) -> None:
    """
    Add or remove requirements from a model's conda.yaml and requirements.txt files.

    The process involves downloading these two files from the model artifacts
    (if they're non-local), updating them with the specified requirements,
    and then overwriting the existing files. Should the artifact repository
    associated with the model artifacts disallow overwriting, this function will
    fail.

    Note that model registry URIs (i.e. URIs in the form ``models:/``) are not
    supported, as artifacts in the model registry are intended to be read-only.

    If adding requirements, the function will overwrite any existing requirements
    that overlap, or else append the new requirements to the existing list.

    If removing requirements, the function will ignore any version specifiers,
    and remove all the specified package names. Any requirements that are not
    found in the existing files will be ignored.

    Args:
        model_uri (str): The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``mlflow-artifacts:/path/to/model``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
            artifact-locations>`_.

        operation (Literal["add", "remove]): The operation to perform.
            Must be one of "add" or "remove".

        requirement_list (List[str]): A list of requirements to add or remove from the model.
            For example: ["numpy==1.20.3", "pandas>=1.3.3"]
    """
    if ModelsArtifactRepository.is_models_uri(model_uri):
        raise MlflowException(f'Failed to set requirements on "{model_uri}". ' + 'Model URIs with the `models:/` scheme are not supported.', INVALID_PARAMETER_VALUE)
    resolved_uri = model_uri
    if RunsArtifactRepository.is_runs_uri(model_uri):
        resolved_uri = RunsArtifactRepository.get_underlying_uri(model_uri)
    _logger.info(f'Retrieving model requirements files from {resolved_uri}...')
    local_paths = get_model_requirements_files(resolved_uri)
    conda_yaml_path = local_paths.conda
    requirements_txt_path = local_paths.requirements
    old_conda_reqs = _get_requirements_from_file(conda_yaml_path)
    old_requirements_reqs = _get_requirements_from_file(requirements_txt_path)
    if operation == 'add':
        updated_conda_reqs = _add_or_overwrite_requirements(requirement_list, old_conda_reqs)
        updated_requirements_reqs = _add_or_overwrite_requirements(requirement_list, old_requirements_reqs)
    else:
        updated_conda_reqs = _remove_requirements(requirement_list, old_conda_reqs)
        updated_requirements_reqs = _remove_requirements(requirement_list, old_requirements_reqs)
    _write_requirements_to_file(conda_yaml_path, updated_conda_reqs)
    _write_requirements_to_file(requirements_txt_path, updated_requirements_reqs)
    _logger.info(f'Done updating requirements!\n\nOld requirements:\n{pformat([str(req) for req in old_conda_reqs])}\n\nUpdated requirements:\n{pformat(updated_conda_reqs)}\n')
    _logger.info(f'Uploading updated requirements files to {resolved_uri}...')
    _upload_artifact_to_uri(conda_yaml_path, resolved_uri)
    _upload_artifact_to_uri(requirements_txt_path, resolved_uri)