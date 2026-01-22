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
def _get_or_create_execution_directory(recipe_root_path: str, recipe_steps: List[BaseStep], template: str) -> str:
    """
    Obtains the path of the execution directory on the local filesystem corresponding to the
    specified recipe, creating the execution directory and its required contents if they do
    not already exist.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.
        recipe_steps: A list of all the steps contained in the specified recipe.
        template: The template to use to generate the makefile.

    Returns:
        The absolute path of the execution directory on the local filesystem for the specified
        recipe.
    """
    execution_dir_path = get_or_create_base_execution_directory(recipe_root_path=recipe_root_path)
    _create_makefile(recipe_root_path, execution_dir_path, template)
    for step in recipe_steps:
        step_output_subdir_path = _get_step_output_directory_path(execution_dir_path, step.name)
        os.makedirs(step_output_subdir_path, exist_ok=True)
    return execution_dir_path