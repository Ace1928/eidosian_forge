import json
import os
import time
from contextlib import contextmanager
from enum import Enum
from typing import NamedTuple, Optional
from unittest import mock
import requests
import mlflow
def _mock_embeddings_response(num_texts):
    return _MockResponse(200, {'object': 'list', 'data': [{'object': 'embedding', 'embedding': [0.0], 'index': i} for i in range(num_texts)], 'model': 'text-embedding-ada-002', 'usage': {'prompt_tokens': 8, 'total_tokens': 8}})