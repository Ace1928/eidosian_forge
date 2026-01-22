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
def _get_experiment_metadata(self):
    project_metadata = self._metadata or {}
    git_info = ls_env.get_git_info()
    if git_info:
        project_metadata = {**project_metadata, 'git': git_info}
    if self._experiment:
        project_metadata = {**self._experiment.metadata, **project_metadata}
    return project_metadata