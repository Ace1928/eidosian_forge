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
def with_evaluators(self, evaluators: Sequence[Union[EVALUATOR_T, RunEvaluator]], *, max_concurrency: Optional[int]=None) -> _ExperimentManager:
    """Lazily apply the provided evaluators to the experiment."""
    evaluators = _resolve_evaluators(evaluators)
    context = copy_context()
    experiment_results = context.run(self._score, evaluators, max_concurrency=max_concurrency)
    r1, r2, r3 = itertools.tee(experiment_results, 3)
    return _ExperimentManager((result['example'] for result in r1), experiment=self._experiment, metadata=self._metadata, client=self.client, runs=(result['run'] for result in r2), evaluation_results=(result['evaluation_results'] for result in r3), summary_results=self._summary_results)