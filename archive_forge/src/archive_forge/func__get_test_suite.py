from __future__ import annotations
import atexit
import concurrent.futures
import datetime
import functools
import inspect
import logging
import threading
import uuid
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, overload
import orjson
from typing_extensions import TypedDict
from langsmith import client as ls_client
from langsmith import env as ls_env
from langsmith import run_helpers as rh
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
def _get_test_suite(client: ls_client.Client, test_suite_name: str) -> ls_schemas.Dataset:
    if client.has_dataset(dataset_name=test_suite_name):
        return client.read_dataset(dataset_name=test_suite_name)
    else:
        repo = ls_env.get_git_info().get('remote_url') or ''
        description = 'Unit test suite'
        if repo:
            description += f' for {repo}'
        return client.create_dataset(dataset_name=test_suite_name, description=description)