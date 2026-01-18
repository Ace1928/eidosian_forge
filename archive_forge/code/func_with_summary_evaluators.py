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
def with_summary_evaluators(self, summary_evaluators: Sequence[SUMMARY_EVALUATOR_T]) -> _ExperimentManager:
    """Lazily apply the provided summary evaluators to the experiment."""
    wrapped_evaluators = _wrap_summary_evaluators(summary_evaluators)
    context = copy_context()
    aggregate_feedback_gen = context.run(self._apply_summary_evaluators, wrapped_evaluators)
    return _ExperimentManager(self.examples, experiment=self._experiment, metadata=self._metadata, client=self.client, runs=self.runs, evaluation_results=self._evaluation_results, summary_results=aggregate_feedback_gen)