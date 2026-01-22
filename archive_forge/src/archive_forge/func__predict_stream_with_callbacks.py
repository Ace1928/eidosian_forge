import contextlib
import functools
import importlib.util
import logging
import os
import sys
import uuid
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Union
import cloudpickle
import pandas as pd
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import _MLFLOW_TESTING
from mlflow.exceptions import MlflowException
from mlflow.langchain._langchain_autolog import (
from mlflow.langchain._rag_utils import _CODE_CONFIG, _CODE_PATH, _set_config_path
from mlflow.langchain.databricks_dependencies import (
from mlflow.langchain.runnables import _load_runnables, _save_runnables
from mlflow.langchain.utils import (
from mlflow.models import Model, ModelInputExample, ModelSignature, get_model_info
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.schema import ColSpec, DataType, Schema
from mlflow.utils.annotations import deprecated, experimental
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _predict_stream_with_callbacks(self, data: Any, params: Optional[Dict[str, Any]]=None, callback_handlers=None, convert_chat_responses=False) -> Iterator[Union[str, Dict[str, Any]]]:
    """
        Args:
            data: Model input data, only single input is allowed.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.
            callback_handlers: Callback handlers to pass to LangChain.
            convert_chat_responses: If true, forcibly convert response to chat model
                response format.

        Returns:
            An iterator of model prediction chunks.
        """
    from mlflow.langchain.api_request_parallel_processor import process_stream_request
    if isinstance(data, list):
        raise MlflowException('LangChain model predict_stream only supports single input.')
    data = _convert_ndarray_to_list(data)
    return process_stream_request(lc_model=self.lc_model, request_json=data, callback_handlers=callback_handlers, convert_chat_responses=convert_chat_responses)