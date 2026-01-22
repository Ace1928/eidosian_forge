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
def _create_makefile(recipe_root_path, execution_directory_path, template) -> None:
    """
    Creates a Makefile with a set of relevant MLflow Recipes targets for the specified recipe,
    overwriting the preexisting Makefile if one exists. The Makefile is created in the specified
    execution directory.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.
        execution_directory_path: The absolute path of the execution directory on the local
            filesystem for the specified recipe. The Makefile is created in this directory.
        template: The template to use to generate the makefile.
    """
    makefile_path = os.path.join(execution_directory_path, 'Makefile')
    if template == 'regression/v1' or template == 'classification/v1':
        makefile_to_use = _MAKEFILE_FORMAT_STRING
        steps_folder_path = os.path.join(recipe_root_path, 'steps')
        if not os.path.exists(steps_folder_path):
            os.mkdir(steps_folder_path)
        for required_file in ['ingest.py', 'split.py', 'train.py', 'transform.py', 'custom_metrics.py']:
            required_file_path = os.path.join(steps_folder_path, required_file)
            if not os.path.exists(required_file_path):
                try:
                    with open(required_file_path, 'w') as f:
                        f.write('# Created by MLflow Pipeliens\n')
                except OSError:
                    pass
            if not os.path.exists(required_file_path):
                raise ValueError(f'Can not find required file {required_file_path} from steps folder. Please create empty python file if the step is not used.')
    else:
        raise ValueError(f'Invalid template: {template}')
    makefile_contents = makefile_to_use.format(path=_MakefilePathFormat(os.path.abspath(recipe_root_path), execution_directory_path=os.path.abspath(execution_directory_path)))
    with open(makefile_path, 'w') as f:
        f.write(makefile_contents)