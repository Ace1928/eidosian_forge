from __future__ import annotations
import os
import time
import uuid
import random
import typing
import logging
import traceback
import contextlib
import contextvars
import asyncio
import functools
import inspect
from concurrent import futures
from lazyops.utils.helpers import import_function, timer, build_batches
from lazyops.utils.ahelpers import amap_v2 as concurrent_map
def _get_jobkey_func():
    global _JobKeyFunc
    if _JobKeyFunc is None:
        from aiokeydb.v2.configs import settings
        _JobKeyFunc = _JobKeyMethod[settings.worker.job_key_method]
    return _JobKeyFunc()