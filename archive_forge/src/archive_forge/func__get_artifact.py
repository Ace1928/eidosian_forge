import abc
import logging
import os
from typing import List, Optional
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.recipes import dag_help_strings
from mlflow.recipes.artifacts import Artifact
from mlflow.recipes.step import BaseStep, StepClass, StepStatus
from mlflow.recipes.utils import (
from mlflow.recipes.utils.execution import (
from mlflow.recipes.utils.step import display_html
from mlflow.utils.class_utils import _get_class_from_string
def _get_artifact(self, artifact_name: str) -> Artifact:
    """
        Read an Artifact object from recipe output. artifact names can be obtained
        from `Recipe.inspect()` or `Recipe.run()` output.

        Returns None if the specified artifact is not found.
        Raise an error if the artifact is not supported.
        """
    for step in self._steps:
        for artifact in step.get_artifacts():
            if artifact.name() == artifact_name:
                return artifact
    raise MlflowException(f"The artifact with name '{artifact_name}' is not supported.", error_code=INVALID_PARAMETER_VALUE)