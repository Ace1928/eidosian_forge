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
def _resolve_run_id(self, run: Union[ls_schemas.Run, ls_schemas.RunBase, str, uuid.UUID], load_child_runs: bool) -> ls_schemas.Run:
    """Resolve the run ID.

        Parameters
        ----------
        run : Run or RunBase or str or UUID
            The run to resolve.
        load_child_runs : bool
            Whether to load child runs.

        Returns:
        -------
        Run
            The resolved run.

        Raises:
        ------
        TypeError
            If the run type is invalid.
        """
    if isinstance(run, (str, uuid.UUID)):
        run_ = self.read_run(run, load_child_runs=load_child_runs)
    else:
        run_ = cast(ls_schemas.Run, run)
    return run_