import abc
import concurrent.futures
import queue
import typing
from typing import Callable, List, Optional
import warnings
def _make_default_thread_pool_executor() -> concurrent.futures.ThreadPoolExecutor:
    return concurrent.futures.ThreadPoolExecutor(max_workers=10, thread_name_prefix='ThreadPoolExecutor-ThreadScheduler')