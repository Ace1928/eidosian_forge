from __future__ import annotations
import pytest
import asyncio
import os
import sys
from functools import partial
from operator import add
from distributed import Client, SchedulerPlugin, WorkerPlugin
from distributed.utils_test import cleanup  # noqa F401
from distributed.utils_test import client as c  # noqa F401
from distributed.utils_test import (  # noqa F401
import dask
import dask.bag as db
from dask import compute, delayed, persist
from dask.base import compute_as_if_collection, get_scheduler
from dask.blockwise import Blockwise
from dask.delayed import Delayed
from dask.distributed import futures_of, wait
from dask.layers import ShuffleLayer, SimpleShuffleLayer
from dask.utils import get_named_args, get_scheduler_lock, tmpdir, tmpfile
from dask.utils_test import inc
@ignore_sync_scheduler_warning
@pytest.mark.parametrize('npartitions', [1, pytest.param(4, marks=pytest.mark.xfail(reason='HDF not multi-process safe', strict=False)), pytest.param(10, marks=pytest.mark.xfail(reason='HDF not multi-process safe', strict=False))])
@pytest.mark.xfail_with_pyarrow_strings
def test_to_hdf_scheduler_distributed(npartitions, c):
    pytest.importorskip('numpy')
    pytest.importorskip('pandas')
    from dask.dataframe.io.tests.test_hdf import test_to_hdf_schedulers
    test_to_hdf_schedulers(None, npartitions)