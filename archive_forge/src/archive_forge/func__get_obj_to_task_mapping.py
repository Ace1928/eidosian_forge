import itertools
import logging
import os
import warnings
from string import Formatter
from typing import Any, Dict, Optional, Set
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import MLFLOW_OPENAI_SECRET_SCOPE
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types import ColSpec, Schema, TensorSpec
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
from mlflow.utils.openai_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _get_obj_to_task_mapping():
    import openai
    if Version(_get_openai_package_version()).major < 1:
        from openai import api_resources as ar
        return {ar.Audio: ar.Audio.OBJECT_NAME, ar.ChatCompletion: ar.ChatCompletion.OBJECT_NAME, ar.Completion: ar.Completion.OBJECT_NAME, ar.Edit: ar.Edit.OBJECT_NAME, ar.Deployment: ar.Deployment.OBJECT_NAME, ar.Embedding: ar.Embedding.OBJECT_NAME, ar.Engine: ar.Engine.OBJECT_NAME, ar.File: ar.File.OBJECT_NAME, ar.Image: ar.Image.OBJECT_NAME, ar.FineTune: ar.FineTune.OBJECT_NAME, ar.Model: ar.Model.OBJECT_NAME, ar.Moderation: 'moderations'}
    else:
        return {openai.audio: 'audio', openai.chat.completions: 'chat.completions', openai.completions: 'completions', openai.images.edit: 'images.edit', openai.embeddings: 'embeddings', openai.files: 'files', openai.images: 'images', openai.fine_tuning: 'fine_tuning', openai.moderations: 'moderations', openai.models: 'models'}