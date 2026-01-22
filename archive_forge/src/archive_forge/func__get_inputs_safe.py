from __future__ import annotations
import asyncio
import contextlib
import contextvars
import datetime
import functools
import inspect
import logging
import traceback
import uuid
import warnings
from contextvars import copy_context
from typing import (
from langsmith import client as ls_client
from langsmith import run_trees, utils
from langsmith._internal import _aiter as aitertools
def _get_inputs_safe(signature: inspect.Signature, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    try:
        return _get_inputs(signature, *args, **kwargs)
    except BaseException as e:
        LOGGER.debug(f'Failed to get inputs for {signature}: {e}')
        return {'args': args, 'kwargs': kwargs}