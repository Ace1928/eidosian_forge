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
class RouteType(str, Enum):
    LLM_V1_COMPLETIONS = 'llm/v1/completions'
    LLM_V1_CHAT = 'llm/v1/chat'
    LLM_V1_EMBEDDINGS = 'llm/v1/embeddings'