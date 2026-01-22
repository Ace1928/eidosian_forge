from __future__ import annotations
import collections
import datetime
import functools
import importlib
import importlib.metadata
import io
import json
import logging
import os
import random
import re
import socket
import sys
import threading
import time
import uuid
import warnings
import weakref
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue, Queue
from typing import (
from urllib import parse as urllib_parse
import orjson
import requests
from requests import adapters as requests_adapters
from urllib3.util import Retry
import langsmith
from langsmith import env as ls_env
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
def _load_child_runs(self, run: ls_schemas.Run) -> ls_schemas.Run:
    """Load child runs for a given run.

        Parameters
        ----------
        run : Run
            The run to load child runs for.

        Returns:
        -------
        Run
            The run with loaded child runs.

        Raises:
        ------
        LangSmithError
            If a child run has no parent.
        """
    child_runs = self.list_runs(id=run.child_run_ids)
    treemap: DefaultDict[uuid.UUID, List[ls_schemas.Run]] = collections.defaultdict(list)
    runs: Dict[uuid.UUID, ls_schemas.Run] = {}
    for child_run in sorted(child_runs, key=lambda r: r.dotted_order):
        if child_run.parent_run_id is None:
            raise ls_utils.LangSmithError(f'Child run {child_run.id} has no parent')
        treemap[child_run.parent_run_id].append(child_run)
        runs[child_run.id] = child_run
    run.child_runs = treemap.pop(run.id, [])
    for run_id, children in treemap.items():
        runs[run_id].child_runs = children
    return run