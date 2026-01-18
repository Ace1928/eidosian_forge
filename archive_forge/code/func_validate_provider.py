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
@validator('provider', pre=True)
def validate_provider(cls, value):
    if isinstance(value, Provider):
        return value
    formatted_value = value.replace('-', '_').upper()
    if formatted_value in Provider.__members__:
        return Provider[formatted_value]
    raise MlflowException.invalid_parameter_value(f"The provider '{value}' is not supported.")