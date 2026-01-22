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
@staticmethod
def _parse_output_lines(output_lines_of_make: List[str]) -> List[str]:
    """
        Parse the output lines of Make to get steps to run.
        """

    def get_step_to_run(output_line: str):
        m = re.search(_ExecutionPlan._MSG_REGEX, output_line)
        return m.group(1) if m else None

    def steps_to_run():
        for output_line in output_lines_of_make:
            step = get_step_to_run(output_line)
            if step is not None:
                yield step
    return list(steps_to_run())