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
def _collect_extra(extra_outer: dict, langsmith_extra: LangSmithExtra) -> dict:
    run_extra = langsmith_extra.get('run_extra', None)
    if run_extra:
        extra_inner = {**extra_outer, **run_extra}
    else:
        extra_inner = extra_outer
    return extra_inner