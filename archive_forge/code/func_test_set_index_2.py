from __future__ import annotations
import contextlib
import itertools
import multiprocessing as mp
import os
import pickle
import random
import string
import tempfile
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from datetime import date, time
from decimal import Decimal
from functools import partial
from unittest import mock
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.base import compute_as_if_collection
from dask.dataframe._compat import (
from dask.dataframe.shuffle import (
from dask.dataframe.utils import assert_eq, make_meta
from dask.optimization import cull
def test_set_index_2(shuffle_method):
    df = dd.demo.make_timeseries('2000', '2004', {'value': float, 'name': str, 'id': int}, freq='2h', partition_freq=f'1{ME}', seed=1)
    df2 = df.set_index('name', shuffle_method=shuffle_method)
    df2.value.sum().compute(scheduler='sync')