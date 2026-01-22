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
def _load_traces(project: Union[str, uuid.UUID], client: langsmith.Client, load_nested: bool=False) -> List[schemas.Run]:
    """Load nested traces for a given project."""
    execution_order = None if load_nested else 1
    if isinstance(project, uuid.UUID) or _is_uuid(project):
        runs = client.list_runs(project_id=project, execution_order=execution_order)
    else:
        runs = client.list_runs(project_name=project, execution_order=execution_order)
    if not load_nested:
        return list(runs)
    treemap: DefaultDict[uuid.UUID, List[schemas.Run]] = collections.defaultdict(list)
    results = []
    all_runs = {}
    for run in runs:
        if run.parent_run_id is not None:
            treemap[run.parent_run_id].append(run)
        else:
            results.append(run)
        all_runs[run.id] = run
    for run_id, child_runs in treemap.items():
        all_runs[run_id].child_runs = sorted(child_runs, key=lambda r: r.dotted_order)
    return results