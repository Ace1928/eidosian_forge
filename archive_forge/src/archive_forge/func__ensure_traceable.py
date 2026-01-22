from __future__ import annotations
import collections
import concurrent.futures as cf
import datetime
import functools
import itertools
import logging
import pathlib
import threading
import uuid
from contextvars import copy_context
from typing import (
from requests import HTTPError
from typing_extensions import TypedDict
import langsmith
from langsmith import env as ls_env
from langsmith import run_helpers as rh
from langsmith import run_trees, schemas
from langsmith import utils as ls_utils
from langsmith.evaluation.evaluator import (
from langsmith.evaluation.integrations import LangChainStringEvaluator
def _ensure_traceable(target: TARGET_T) -> rh.SupportsLangsmithExtra:
    """Ensure the target function is traceable."""
    if not callable(target):
        raise ValueError('Target must be a callable function.')
    if rh.is_traceable_function(target):
        fn = cast(rh.SupportsLangsmithExtra, target)
    else:
        fn = rh.traceable(name='Target')(target)
    return fn