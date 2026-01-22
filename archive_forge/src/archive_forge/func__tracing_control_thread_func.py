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
def _tracing_control_thread_func(client_ref: weakref.ref[Client]) -> None:
    client = client_ref()
    if client is None:
        return
    tracing_queue = client.tracing_queue
    assert tracing_queue is not None
    batch_ingest_config = _ensure_ingest_config(client.info)
    size_limit: int = batch_ingest_config['size_limit']
    scale_up_nthreads_limit: int = batch_ingest_config['scale_up_nthreads_limit']
    scale_up_qsize_trigger: int = batch_ingest_config['scale_up_qsize_trigger']
    sub_threads: List[threading.Thread] = []
    num_known_refs = 3
    while threading.main_thread().is_alive() and sys.getrefcount(client) > num_known_refs + len(sub_threads):
        for thread in sub_threads:
            if not thread.is_alive():
                sub_threads.remove(thread)
        if len(sub_threads) < scale_up_nthreads_limit and tracing_queue.qsize() > scale_up_qsize_trigger:
            new_thread = threading.Thread(target=_tracing_sub_thread_func, args=(weakref.ref(client),))
            sub_threads.append(new_thread)
            new_thread.start()
        if (next_batch := _tracing_thread_drain_queue(tracing_queue, limit=size_limit)):
            _tracing_thread_handle_batch(client, tracing_queue, next_batch)
    while (next_batch := _tracing_thread_drain_queue(tracing_queue, limit=size_limit, block=False)):
        _tracing_thread_handle_batch(client, tracing_queue, next_batch)