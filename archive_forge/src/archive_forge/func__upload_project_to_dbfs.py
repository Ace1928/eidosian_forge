import hashlib
import json
import logging
import os
import posixpath
import re
import tempfile
import textwrap
import time
from shlex import quote
from mlflow import tracking
from mlflow.entities import RunStatus
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID, MLFLOW_TRACKING_URI
from mlflow.exceptions import ExecutionException, MlflowException
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.projects.utils import MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils import databricks_utils, file_utils, rest_utils
from mlflow.utils.mlflow_tags import (
from mlflow.utils.uri import is_databricks_uri, is_http_uri
from mlflow.version import VERSION, is_release_version
def _upload_project_to_dbfs(self, project_dir, experiment_id):
    """
        Tars a project directory into an archive in a temp dir and uploads it to DBFS, returning
        the HDFS-style URI of the tarball in DBFS (e.g. dbfs:/path/to/tar).

        Args:
            project_dir: Path to a directory containing an MLflow project to upload to DBFS (e.g.
                a directory containing an MLproject file).
        """
    with tempfile.TemporaryDirectory() as temp_tarfile_dir:
        temp_tar_filename = os.path.join(temp_tarfile_dir, 'project.tar.gz')

        def custom_filter(x):
            return None if os.path.basename(x.name) == 'mlruns' else x
        directory_size = file_utils._get_local_project_dir_size(project_dir)
        _logger.info(f'=== Creating tarball from {project_dir} in temp directory {temp_tarfile_dir} ===')
        _logger.info(f'=== Total file size to compress: {directory_size} KB ===')
        file_utils.make_tarfile(temp_tar_filename, project_dir, DB_TARFILE_ARCHIVE_NAME, custom_filter=custom_filter)
        with open(temp_tar_filename, 'rb') as tarred_project:
            tarfile_hash = hashlib.sha256(tarred_project.read()).hexdigest()
        dbfs_path = posixpath.join(DBFS_EXPERIMENT_DIR_BASE, str(experiment_id), 'projects-code', f'{tarfile_hash}.tar.gz')
        tar_size = file_utils._get_local_file_size(temp_tar_filename)
        dbfs_fuse_uri = posixpath.join('/dbfs', dbfs_path)
        if not self._dbfs_path_exists(dbfs_path):
            _logger.info(f'=== Uploading project tarball (size: {tar_size} KB) to {dbfs_fuse_uri} ===')
            self._upload_to_dbfs(temp_tar_filename, dbfs_fuse_uri)
            _logger.info('=== Finished uploading project to %s ===', dbfs_fuse_uri)
        else:
            _logger.info('=== Project already exists in DBFS ===')
    return dbfs_fuse_uri