import inspect
import logging
import os
import shutil
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import cloudpickle
import yaml
import mlflow.pyfunc
import mlflow.utils
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _extract_type_hints
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.llm import ChatMessage, ChatParams, ChatResponse
from mlflow.utils.annotations import experimental
from mlflow.utils.environment import (
from mlflow.utils.file_utils import TempDir, get_total_file_size, write_to
from mlflow.utils.model_utils import _get_flavor_configuration, _validate_and_copy_code_paths
from mlflow.utils.requirements_utils import _get_pinned_requirement
@experimental
class ChatModel(PythonModel, metaclass=ABCMeta):
    """
    A subclass of :class:`~PythonModel` that makes it more convenient to implement models
    that are compatible with popular LLM chat APIs. By subclassing :class:`~ChatModel`,
    users can create MLflow models with a ``predict()`` method that is more convenient
    for chat tasks than the generic :class:`~PythonModel` API. ChatModels automatically
    define input/output signatures and an input example, so manually specifying these values
    when calling :func:`mlflow.pyfunc.save_model() <mlflow.pyfunc.save_model>` is not necessary.

    See the documentation of the ``predict()`` method below for details on that parameters and
    outputs that are expected by the ``ChatModel`` API.
    """

    @abstractmethod
    def predict(self, context, messages: List[ChatMessage], params: ChatParams) -> ChatResponse:
        """
        Evaluates a chat input and produces a chat output.

        Args:
            messages (List[:py:class:`ChatMessage <mlflow.types.llm.ChatMessage>`]):
                A list of :py:class:`ChatMessage <mlflow.types.llm.ChatMessage>`
                objects representing chat history.
            params (:py:class:`ChatParams <mlflow.types.llm.ChatParams>`):
                A :py:class:`ChatParams <mlflow.types.llm.ChatParams>` object
                containing various parameters used to modify model behavior during
                inference.

        Returns:
            A :py:class:`ChatResponse <mlflow.types.llm.ChatResponse>` object containing
            the model's response(s), as well as other metadata.
        """