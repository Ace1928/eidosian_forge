import hashlib
import logging
import os
import pathlib
import re
import shutil
from typing import Dict, List
from mlflow.environment_variables import (
from mlflow.recipes.step import BaseStep, StepStatus
from mlflow.utils.file_utils import read_yaml, write_yaml
from mlflow.utils.process import _exec_cmd
def _write_updated_step_confs(recipe_steps: List[BaseStep], execution_directory_path: str) -> None:
    """
    Compares the in-memory configuration state of the specified recipe steps with step-specific
    internal configuration files written by prior executions. If updates are found, writes updated
    state to the corresponding files. If no updates are found, configuration state is not
    rewritten.

    Args:
        recipe_steps: A list of all the steps contained in the specified recipe.
        execution_directory_path: The absolute path of the execution directory on the local
            filesystem for the specified recipe. Configuration files are written to step-specific
            subdirectories of this execution directory.
    """
    for step in recipe_steps:
        step_subdir_path = os.path.join(execution_directory_path, _STEPS_SUBDIRECTORY_NAME, step.name)
        step_conf_path = os.path.join(step_subdir_path, _STEP_CONF_YAML_NAME)
        if os.path.exists(step_conf_path):
            prev_step_conf = read_yaml(root=step_subdir_path, file_name=_STEP_CONF_YAML_NAME)
        else:
            prev_step_conf = None
        if prev_step_conf != step.step_config:
            write_yaml(root=step_subdir_path, file_name=_STEP_CONF_YAML_NAME, data=step.step_config, overwrite=True, sort_keys=True)