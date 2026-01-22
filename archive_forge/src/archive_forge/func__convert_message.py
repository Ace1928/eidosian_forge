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
def _convert_message(message: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract message from a message object."""
    message_type = _get_message_type(message)
    message_data = _get_message_fields(message)
    return {'type': message_type, 'data': message_data}