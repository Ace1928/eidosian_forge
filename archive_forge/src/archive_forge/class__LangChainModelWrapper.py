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
class _LangChainModelWrapper:

    def __init__(self, lc_model):
        self.lc_model = lc_model

    def predict(self, data: Union[pd.DataFrame, List[Union[str, Dict[str, Any]]], Any], params: Optional[Dict[str, Any]]=None) -> List[Union[str, Dict[str, Any]]]:
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.

        Returns:
            Model predictions.
        """
        from mlflow.langchain.api_request_parallel_processor import process_api_requests
        messages, return_first_element = self._prepare_messages(data)
        results = process_api_requests(lc_model=self.lc_model, requests=messages)
        return results[0] if return_first_element else results

    @experimental
    def _predict_with_callbacks(self, data: Union[pd.DataFrame, List[Union[str, Dict[str, Any]]], Any], params: Optional[Dict[str, Any]]=None, callback_handlers=None, convert_chat_responses=False) -> List[Union[str, Dict[str, Any]]]:
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.
            callback_handlers: Callback handlers to pass to LangChain.
            convert_chat_responses: If true, forcibly convert response to chat model
                response format.

        Returns:
            Model predictions.
        """
        from mlflow.langchain.api_request_parallel_processor import process_api_requests
        messages, return_first_element = self._prepare_messages(data)
        results = process_api_requests(lc_model=self.lc_model, requests=messages, callback_handlers=callback_handlers, convert_chat_responses=convert_chat_responses)
        return results[0] if return_first_element else results

    def _prepare_messages(self, data):
        """
        Return a tuple of (preprocessed_data, return_first_element)
        `preprocessed_data` is always a list,
        and `return_first_element` means if True, we should return the first element
        of inference result, otherwise we should return the whole inference result.
        """
        if isinstance(data, pd.DataFrame):
            if list(data.columns) == [0]:
                data = data.to_dict('list')[0]
            else:
                data = data.to_dict(orient='records')
        data = _convert_ndarray_to_list(data)
        if not isinstance(data, list):
            return ([data], True)
        if isinstance(data, list):
            return (data, False)
        raise mlflow.MlflowException.invalid_parameter_value(f'Input must be a pandas DataFrame or a list for model {self.lc_model.__class__.__name__}')

    def predict_stream(self, data: Any, params: Optional[Dict[str, Any]]=None) -> Iterator[Union[str, Dict[str, Any]]]:
        """
        Args:
            data: Model input data, only single input is allowed.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.

        Returns:
            An iterator of model prediction chunks.
        """
        from mlflow.langchain.api_request_parallel_processor import process_stream_request
        if isinstance(data, list):
            raise MlflowException('LangChain model predict_stream only supports single input.')
        data = _convert_ndarray_to_list(data)
        return process_stream_request(lc_model=self.lc_model, request_json=data)

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