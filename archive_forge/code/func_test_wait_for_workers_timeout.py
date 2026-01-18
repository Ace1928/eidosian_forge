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
from distributed.utils_test import cluster, inc  # noqa: E402
def test_wait_for_workers_timeout():
    cluster = LocalCluster(n_workers=0, processes=False, threads_per_worker=2)
    client = Client(cluster)
    try:
        with parallel_config(backend='dask', wait_for_workers_timeout=0.1):
            msg = 'DaskDistributedBackend has no worker after 0.1 seconds.'
            with pytest.raises(TimeoutError, match=msg):
                Parallel()((delayed(inc)(i) for i in range(10)))
        with parallel_config(backend='dask', wait_for_workers_timeout=0):
            msg = 'DaskDistributedBackend has no active worker'
            with pytest.raises(RuntimeError, match=msg):
                Parallel()((delayed(inc)(i) for i in range(10)))
    finally:
        client.close()
        cluster.close()