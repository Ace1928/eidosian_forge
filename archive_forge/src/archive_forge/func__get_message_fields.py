import contextlib
import enum
import functools
import logging
import os
import pathlib
import subprocess
import threading
from typing import (
import requests
from urllib3.util import Retry
from langsmith import schemas as ls_schemas
def _get_message_fields(message: Mapping[str, Any]) -> Mapping[str, Any]:
    if not message:
        raise ValueError('Message is empty.')
    if 'lc' in message:
        if 'kwargs' not in message:
            raise ValueError(f'Unexpected format for serialized message: {message} Message does not have kwargs.')
        return message['kwargs']
    else:
        if 'data' not in message:
            raise ValueError(f'Unexpected format for stored message: {message} Message does not have data.')
        return message['data']