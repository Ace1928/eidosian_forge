from __future__ import annotations
import copy
import os
import re
import sys
from functools import partial
from operator import add, neg
import pytest
from dask.dot import _to_cytoscape_json, cytoscape_graph
from dask import delayed
from dask.utils import ensure_not_exists
def test_task_label():
    assert task_label((partial(add, 1), 1)) == 'add'
    assert task_label((add, 1)) == 'add'
    assert task_label((add, (add, 1, 2))) == 'add(...)'