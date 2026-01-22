import base64
import functools
import inspect
import json
import logging
import posixpath
import re
import textwrap
import warnings
from typing import Any, AsyncGenerator, List, Optional
from urllib.parse import urlparse
from starlette.responses import StreamingResponse
from mlflow.environment_variables import MLFLOW_GATEWAY_URI
from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import MLFLOW_AI_GATEWAY_MOSAICML_CHAT_SUPPORTED_MODEL_PREFIXES
from mlflow.utils.uri import append_to_uri_path
def assemble_uri_path(paths: List[str]) -> str:
    """Assemble a correct URI path from a list of path parts.

    Args:
        paths: A list of strings representing parts of a URI path.

    Returns:
        A string representing the complete assembled URI path.

    """
    stripped_paths = [path.strip('/').lstrip('/') for path in paths if path]
    return '/' + posixpath.join(*stripped_paths) if stripped_paths else '/'