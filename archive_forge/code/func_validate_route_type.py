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
@validator('route_type', pre=True)
def validate_route_type(cls, value):
    if value in RouteType._value2member_map_:
        return value
    raise MlflowException.invalid_parameter_value(f"The route_type '{value}' is not supported.")