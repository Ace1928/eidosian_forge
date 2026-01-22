import json
import os
import time
from contextlib import contextmanager
from enum import Enum
from typing import NamedTuple, Optional
from unittest import mock
import requests
import mlflow
class _MockResponse:

    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self.content = json.dumps(json_data).encode()
        self.headers = {'Content-Type': 'application/json'}
        self.text = mlflow.__version__
        self.json_data = json_data

    def json(self):
        return self.json_data