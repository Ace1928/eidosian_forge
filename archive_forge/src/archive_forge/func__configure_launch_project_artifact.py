import json
import os
import shutil
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional
from wandb_gql import gql
import wandb
from wandb import util
from wandb.apis import public
from wandb.apis.normalize import normalize_exceptions
from wandb.errors import CommError
from wandb.sdk.artifacts.artifact_state import ArtifactState
from wandb.sdk.data_types._dtypes import InvalidType, Type, TypeRegistry
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.utils import (
def _configure_launch_project_artifact(self, launch_project):
    artifact_string = self._job_info.get('source', {}).get('artifact')
    if artifact_string is None:
        raise LaunchError(f'Job {self.name} had no source artifact')
    code_artifact = self._get_code_artifact(artifact_string)
    launch_project.python_version = self._job_info.get('runtime')
    shutil.copy(self._requirements_file, launch_project.project_dir)
    code_artifact.download(launch_project.project_dir)
    if self._notebook_job:
        self._configure_launch_project_notebook(launch_project)
    else:
        launch_project.set_entry_point(self._entrypoint)