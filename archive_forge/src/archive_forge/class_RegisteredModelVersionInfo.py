import json
import logging
import os
from abc import ABC, abstractmethod
import mlflow
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.tracking import MlflowClient
from mlflow.tracking._tracking_service.utils import _use_tracking_uri
from mlflow.utils.file_utils import chdir
class RegisteredModelVersionInfo:
    _KEY_REGISTERED_MODEL_NAME = 'registered_model_name'
    _KEY_REGISTERED_MODEL_VERSION = 'registered_model_version'

    def __init__(self, name: str, version: int):
        self.name = name
        self.version = version

    def to_json(self, path):
        registered_model_info_dict = {RegisteredModelVersionInfo._KEY_REGISTERED_MODEL_NAME: self.name, RegisteredModelVersionInfo._KEY_REGISTERED_MODEL_VERSION: self.version}
        with open(path, 'w') as f:
            json.dump(registered_model_info_dict, f)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            registered_model_info_dict = json.load(f)
        return cls(name=registered_model_info_dict[RegisteredModelVersionInfo._KEY_REGISTERED_MODEL_NAME], version=registered_model_info_dict[RegisteredModelVersionInfo._KEY_REGISTERED_MODEL_VERSION])