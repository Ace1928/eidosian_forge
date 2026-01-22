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
def _get_message_type(message: Mapping[str, Any]) -> str:
    if not message:
        raise ValueError('Message is empty.')
    if 'lc' in message:
        if 'id' not in message:
            raise ValueError(f'Unexpected format for serialized message: {message} Message does not have an id.')
        return message['id'][-1].replace('Message', '').lower()
    else:
        if 'type' not in message:
            raise ValueError(f'Unexpected format for stored message: {message} Message does not have a type.')
        return message['type']