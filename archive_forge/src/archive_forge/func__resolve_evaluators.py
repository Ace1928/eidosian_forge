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
def _resolve_evaluators(evaluators: Sequence[Union[EVALUATOR_T, RunEvaluator, AEVALUATOR_T]]) -> Sequence[RunEvaluator]:
    results = []
    for evaluator in evaluators:
        if isinstance(evaluator, RunEvaluator):
            results.append(evaluator)
        elif isinstance(evaluator, LangChainStringEvaluator):
            results.append(evaluator.as_run_evaluator())
        else:
            results.append(run_evaluator(evaluator))
    return results