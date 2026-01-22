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
class TinyDAG:
    """Tiny DAG

    Bases on the Kahn's algorithm, and enables parallel visiting of the nodes
    (parallel execution of the workflow items).
    """

    def __init__(self, data=None):
        self._reset()
        self._lock = threading.Lock()
        if data and isinstance(data, dict):
            self.from_dict(data)

    def _reset(self):
        self._graph = dict()
        self._wait_timeout = 120

    @property
    def graph(self):
        """Get graph as adjacency dict"""
        return self._graph

    def add_node(self, node):
        self._graph.setdefault(node, set())

    def add_edge(self, u, v):
        self._graph[u].add(v)

    def from_dict(self, data):
        self._reset()
        for k, v in data.items():
            self.add_node(k)
            for dep in v:
                self.add_edge(k, dep)

    def walk(self, timeout=None):
        """Start the walking from the beginning."""
        if timeout:
            self._wait_timeout = timeout
        return self

    def __iter__(self):
        self._start_traverse()
        return self

    def __next__(self):
        if self._it_cnt > 0:
            self._it_cnt -= 1
            try:
                res = self._queue.get(block=True, timeout=self._wait_timeout)
                return res
            except queue.Empty:
                raise exceptions.SDKException('Timeout waiting for cleanup task to complete')
        else:
            raise StopIteration

    def node_done(self, node):
        """Mark node as "processed" and put following items into the queue"""
        self._done.add(node)
        for v in self._graph[node]:
            self._run_in_degree[v] -= 1
            if self._run_in_degree[v] == 0:
                self._queue.put(v)

    def _start_traverse(self):
        """Initialize graph traversing"""
        self._run_in_degree = self._get_in_degree()
        self._queue: queue.Queue[str] = queue.Queue()
        self._done = set()
        self._it_cnt = len(self._graph)
        for k, v in self._run_in_degree.items():
            if v == 0:
                self._queue.put(k)

    def _get_in_degree(self):
        """Calculate the in_degree (count incoming) for nodes"""
        _in_degree: ty.Dict[str, int] = {u: 0 for u in self._graph.keys()}
        for u in self._graph:
            for v in self._graph[u]:
                _in_degree[v] += 1
        return _in_degree

    def topological_sort(self):
        """Return the graph nodes in the topological order"""
        result = []
        for node in self:
            result.append(node)
            self.node_done(node)
        return result

    def size(self):
        return len(self._graph.keys())

    def is_complete(self):
        return len(self._done) == self.size()