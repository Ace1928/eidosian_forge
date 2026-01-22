from collections.abc import Mapping
import hashlib
import queue
import string
import threading
import time
import typing as ty
import keystoneauth1
from keystoneauth1 import adapter as ks_adapter
from keystoneauth1 import discover
from openstack import _log
from openstack import exceptions
def _get_in_degree(self):
    """Calculate the in_degree (count incoming) for nodes"""
    _in_degree: ty.Dict[str, int] = {u: 0 for u in self._graph.keys()}
    for u in self._graph:
        for v in self._graph[u]:
            _in_degree[v] += 1
    return _in_degree