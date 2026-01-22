import json
import os
import time
from contextlib import contextmanager
from enum import Enum
from typing import NamedTuple, Optional
from unittest import mock
import requests
import mlflow
class _OpenAIEnvVar(str, Enum):
    OPENAI_API_TYPE = 'OPENAI_API_TYPE'
    OPENAI_API_BASE = 'OPENAI_API_BASE'
    OPENAI_API_KEY = 'OPENAI_API_KEY'
    OPENAI_API_KEY_PATH = 'OPENAI_API_KEY_PATH'
    OPENAI_API_VERSION = 'OPENAI_API_VERSION'
    OPENAI_ORGANIZATION = 'OPENAI_ORGANIZATION'
    OPENAI_ENGINE = 'OPENAI_ENGINE'
    OPENAI_DEPLOYMENT_NAME = 'OPENAI_DEPLOYMENT_NAME'

    @property
    def secret_key(self):
        return self.value.lower()

    @classmethod
    def read_environ(cls):
        env_vars = {}
        for e in _OpenAIEnvVar:
            if (value := os.getenv(e.value)):
                env_vars[e.value] = value
        return env_vars