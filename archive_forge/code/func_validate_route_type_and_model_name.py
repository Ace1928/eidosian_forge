import json
import logging
import os
import pathlib
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pydantic
import yaml
from packaging import version
from packaging.version import Version
from pydantic import ConfigDict, Field, ValidationError, root_validator, validator
from pydantic.json import pydantic_encoder
from mlflow.exceptions import MlflowException
from mlflow.gateway.base_models import ConfigModel, LimitModel, ResponseModel
from mlflow.gateway.constants import (
from mlflow.gateway.utils import (
@root_validator(skip_on_failure=True)
def validate_route_type_and_model_name(cls, values):
    route_type = values.get('route_type')
    model = values.get('model')
    if model and model.provider == 'mosaicml' and (route_type == RouteType.LLM_V1_CHAT) and (not is_valid_mosiacml_chat_model(model.name)):
        raise MlflowException.invalid_parameter_value(f"An invalid model has been specified for the chat route. '{model.name}'. Ensure the model selected starts with one of: {MLFLOW_AI_GATEWAY_MOSAICML_CHAT_SUPPORTED_MODEL_PREFIXES}")
    if model and model.provider == 'ai21labs' and (not is_valid_ai21labs_model(model.name)):
        raise MlflowException.invalid_parameter_value(f"An Unsupported AI21Labs model has been specified: '{model.name}'. Please see documentation for supported models.")
    return values