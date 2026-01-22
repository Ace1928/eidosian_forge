import logging
import os
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Union
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.tracking import DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracking._tracking_service.registry import TrackingStoreRegistry
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.uri import _DATABRICKS_UNITY_CATALOG_SCHEME
def _get_git_url_if_present(uri):
    """Return the path git_uri#sub_directory if the URI passed is a local path that's part of
    a Git repo, or returns the original URI otherwise.

    Args:
        uri: The expanded uri.

    Returns:
        The git_uri#sub_directory if the uri is part of a Git repo, otherwise return the original
        uri.
    """
    if '#' in uri:
        return uri
    try:
        from git import GitCommandNotFound, InvalidGitRepositoryError, NoSuchPathError, Repo
    except ImportError as e:
        _logger.warning('Failed to import Git (the git executable is probably not on your PATH), so Git SHA is not available. Error: %s', e)
        return uri
    try:
        repo = Repo(uri, search_parent_directories=True)
        repo_url = f'file://{repo.working_tree_dir}'
        rlpath = uri.replace(repo.working_tree_dir, '')
        if rlpath == '':
            git_path = repo_url
        elif rlpath[0] == '/':
            git_path = repo_url + '#' + rlpath[1:]
        else:
            git_path = repo_url + '#' + rlpath
        return git_path
    except (InvalidGitRepositoryError, GitCommandNotFound, ValueError, NoSuchPathError):
        return uri