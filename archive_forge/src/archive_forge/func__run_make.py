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
def _run_make(execution_directory_path, rule_name: str, extra_env: Dict[str, str], recipe_steps: List[BaseStep]) -> None:
    """
    Runs the specified recipe rule with Make. This method assumes that a Makefile named `Makefile`
    exists in the specified execution directory.

    Args:
        execution_directory_path: The absolute path of the execution directory on the local
            filesystem for the relevant recipe. The Makefile is created in this directory.
        extra_env: Extra environment variables to be defined when running the Make child process.
        rule_name: The name of the Make rule to run.
        recipe_steps: A list of step instances that is a subgraph containing the step specified
            by `rule_name`.
    """
    process = _exec_cmd(['make', '-n', '-f', 'Makefile', rule_name], capture_output=False, stream_output=True, synchronous=False, throw_on_error=False, cwd=execution_directory_path, extra_env=extra_env)
    output_lines = list(iter(process.stdout.readline, ''))
    process.communicate()
    return_code = process.poll()
    if return_code == 0:
        recipe_step_names = [step.name for step in recipe_steps]
        _ExecutionPlan(rule_name, output_lines, recipe_step_names).print()
    _exec_cmd(['make', '-s', '-f', 'Makefile', rule_name], capture_output=False, stream_output=True, synchronous=True, throw_on_error=False, cwd=execution_directory_path, extra_env=extra_env)