import json
import os
import time
from contextlib import contextmanager
from enum import Enum
from typing import NamedTuple, Optional
from unittest import mock
import requests
import mlflow
def _chat_completion_json_sample(content):
    return {'id': 'chatcmpl-123', 'object': 'chat.completion', 'created': 1677652288, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop', 'text': content}], 'usage': {'prompt_tokens': 9, 'completion_tokens': 12, 'total_tokens': 21}}