import json
import os
import sys
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils._capture_modules import (
class _CaptureImportedModulesForHF(_CaptureImportedModules):
    """
    A context manager to capture imported modules by temporarily applying a patch to
    `builtins.__import__` and `importlib.import_module`.
    Used for 'transformers' flavor only.
    """

    def __init__(self, module_to_throw):
        super().__init__()
        self.module_to_throw = module_to_throw

    def _record_imported_module(self, full_module_name):
        if full_module_name == self.module_to_throw or full_module_name.startswith(f'{self.module_to_throw}.'):
            raise ImportError(f'Disabled package {full_module_name}')
        return super()._record_imported_module(full_module_name)