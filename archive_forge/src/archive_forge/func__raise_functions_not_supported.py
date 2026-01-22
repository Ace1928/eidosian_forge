import logging
import warnings
from typing import Any, Dict, List, Mapping, Optional
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import (
from langchain_core.pydantic_v1 import BaseModel, Extra
@staticmethod
def _raise_functions_not_supported() -> None:
    raise ValueError('Function messages are not supported by the MLflow AI Gateway. Please create a feature request at https://github.com/mlflow/mlflow/issues.')