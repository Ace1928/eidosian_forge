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
@staticmethod
def _insert_runtime_env(runs: Sequence[dict]) -> None:
    runtime_env = ls_env.get_runtime_and_metrics()
    for run_create in runs:
        run_extra = cast(dict, run_create.setdefault('extra', {}))
        runtime: dict = run_extra.setdefault('runtime', {})
        run_extra['runtime'] = {**runtime_env, **runtime}
        metadata: dict = run_extra.setdefault('metadata', {})
        langchain_metadata = ls_env.get_langchain_env_var_metadata()
        metadata.update({k: v for k, v in langchain_metadata.items() if k not in metadata})