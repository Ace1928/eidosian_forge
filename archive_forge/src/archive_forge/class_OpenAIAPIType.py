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
class OpenAIAPIType(str, Enum):
    OPENAI = 'openai'
    AZURE = 'azure'
    AZUREAD = 'azuread'

    @classmethod
    def _missing_(cls, value):
        """
        Implements case-insensitive matching of API type strings
        """
        for api_type in cls:
            if api_type.value == value.lower():
                return api_type
        raise MlflowException.invalid_parameter_value(f"Invalid OpenAI API type '{value}'")