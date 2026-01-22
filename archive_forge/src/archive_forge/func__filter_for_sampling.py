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
def _filter_for_sampling(self, runs: Iterable[dict], *, patch: bool=False) -> list[dict]:
    if self.tracing_sample_rate is None:
        return list(runs)
    if patch:
        sampled = []
        for run in runs:
            run_id = _as_uuid(run['id'])
            if run_id in self._sampled_post_uuids:
                sampled.append(run)
                self._sampled_post_uuids.remove(run_id)
        return sampled
    else:
        sampled = []
        for run in runs:
            if random.random() < self.tracing_sample_rate:
                sampled.append(run)
                self._sampled_post_uuids.add(_as_uuid(run['id']))
        return sampled