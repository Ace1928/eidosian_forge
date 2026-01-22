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
class _ExecutionPlan:
    _MSG_REGEX = '^echo "Run MLflow Recipe step: (\\w+)"\\n$'
    _FORMAT_STEPS_CACHED = '%s: No changes. Skipping.'

    def __init__(self, rule_name, output_lines_of_make: List[str], recipe_step_names: List[str]):
        steps_to_run = self._parse_output_lines(output_lines_of_make)
        self.steps_cached = self._infer_cached_steps(rule_name, steps_to_run, recipe_step_names)

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

    @staticmethod
    def _infer_cached_steps(rule_name, steps_to_run, recipe_step_names) -> List[str]:
        """
        Infer cached steps.

        Args:
            rule_name: The name of the Make rule to run.
            steps_to_run: The step names obtained by parsing the Make output showing
                which steps will be executed.
            recipe_step_names: A list of all the step names contained in the specified
                recipe sorted by the execution order.

        """
        index = recipe_step_names.index(rule_name)
        if index == 0:
            return []
        if len(steps_to_run) == 0:
            return recipe_step_names[:index + 1]
        first_step_index = min([recipe_step_names.index(step) for step in steps_to_run])
        return recipe_step_names[:first_step_index]

    def print(self) -> None:
        if len(self.steps_cached) > 0:
            steps_cached_str = ', '.join(self.steps_cached)
            _logger.info(self._FORMAT_STEPS_CACHED, steps_cached_str)