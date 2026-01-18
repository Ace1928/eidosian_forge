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
def to_route(self) -> 'Route':
    return Route(name=self.name, route_type=self.route_type, model=RouteModelInfo(name=self.model.name, provider=self.model.provider), route_url=f'{MLFLOW_GATEWAY_ROUTE_BASE}{self.name}{MLFLOW_QUERY_SUFFIX}', limit=self.limit)