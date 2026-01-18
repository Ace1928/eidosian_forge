from __future__ import print_function, division, absolute_import
import os
import warnings
import pytest
from random import random
from uuid import uuid4
from time import sleep
from .. import Parallel, delayed, parallel_config
from ..parallel import ThreadingBackend, AutoBatchingMixin
from .._dask import DaskDistributedBackend
from distributed import Client, LocalCluster, get_client  # noqa: E402
from distributed.metrics import time  # noqa: E402
from distributed.utils_test import cluster, inc, cleanup  # noqa: E402, F401
def test_scheduler_tasks_cleanup(loop):
    with Client(processes=False, loop=loop) as client:
        with parallel_config(backend='dask'):
            Parallel()((delayed(inc)(i) for i in range(10)))
        start = time()
        while client.cluster.scheduler.tasks:
            sleep(0.01)
            assert time() < start + 5
        assert not client.futures