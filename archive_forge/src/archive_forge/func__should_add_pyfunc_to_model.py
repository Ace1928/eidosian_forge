from __future__ import annotations
import ast
import base64
import binascii
import contextlib
import copy
import functools
import importlib
import json
import logging
import os
import pathlib
import re
import shutil
import string
import sys
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import yaml
from packaging.version import Version
from mlflow import pyfunc
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.models import (
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import (
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _get_root_uri_and_artifact_path
from mlflow.transformers.flavor_config import (
from mlflow.transformers.hub_utils import is_valid_hf_repo_id
from mlflow.transformers.llm_inference_utils import (
from mlflow.transformers.model_io import (
from mlflow.transformers.peft import (
from mlflow.transformers.signature import (
from mlflow.transformers.torch_utils import _TORCH_DTYPE_KEY, _deserialize_torch_dtype
from mlflow.types.utils import _validate_input_dictionary_contains_only_strings_and_lists_of_strings
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import (
from mlflow.utils.environment import (
from mlflow.utils.file_utils import TempDir, get_total_file_size, write_to
from mlflow.utils.logging_utils import suppress_logs
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _should_add_pyfunc_to_model(pipeline) -> bool:
    """
    Discriminator for determining whether a particular task type and model instance from within
    a ``Pipeline`` is currently supported for the pyfunc flavor.

    Image and Video pipelines can still be logged and used, but are not available for
    loading as pyfunc.
    Similarly, esoteric model types (Graph Models, Timeseries Models, and Reinforcement Learning
    Models) are not permitted for loading as pyfunc due to the complex input types that, in
    order to support, will require significant modifications (breaking changes) to the pyfunc
    contract.
    """
    import transformers
    exclusion_model_types = {'GraphormerPreTrainedModel', 'InformerPreTrainedModel', 'TimeSeriesTransformerPreTrainedModel', 'DecisionTransformerPreTrainedModel'}
    exclusion_pipeline_types = ['DocumentQuestionAnsweringPipeline', 'ImageToTextPipeline', 'VisualQuestionAnsweringPipeline', 'ImageSegmentationPipeline', 'DepthEstimationPipeline', 'ObjectDetectionPipeline', 'VideoClassificationPipeline', 'ZeroShotImageClassificationPipeline', 'ZeroShotObjectDetectionPipeline', 'ZeroShotAudioClassificationPipeline']
    for model_type in exclusion_model_types:
        if hasattr(transformers, model_type):
            if isinstance(pipeline.model, getattr(transformers, model_type)):
                return False
    if type(pipeline).__name__ in exclusion_pipeline_types:
        return False
    return True