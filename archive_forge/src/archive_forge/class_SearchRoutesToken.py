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
class SearchRoutesToken:

    def __init__(self, index: int):
        self._index = index

    @property
    def index(self):
        return self._index

    @classmethod
    def decode(cls, encoded_token: str):
        try:
            decoded_token = base64.b64decode(encoded_token)
            parsed_token = json.loads(decoded_token)
            index = int(parsed_token.get('index'))
        except Exception as e:
            raise MlflowException.invalid_parameter_value(f'Invalid SearchRoutes token: {encoded_token}. The index is not defined as a value that can be represented as a positive integer.') from e
        if index < 0:
            raise MlflowException.invalid_parameter_value(f'Invalid SearchRoutes token: {encoded_token}. The index cannot be negative.')
        return cls(index=index)

    def encode(self) -> str:
        token_json = json.dumps({'index': self.index})
        encoded_token_bytes = base64.b64encode(bytes(token_json, 'utf-8'))
        return encoded_token_bytes.decode('utf-8')