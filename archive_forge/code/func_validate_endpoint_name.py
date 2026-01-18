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
@validator('name')
def validate_endpoint_name(cls, route_name):
    if not is_valid_endpoint_name(route_name):
        raise MlflowException.invalid_parameter_value(f"The route name provided contains disallowed characters for a url endpoint. '{route_name}' is invalid. Names cannot contain spaces or any non alphanumeric characters other than hyphen and underscore.")
    return route_name