import concurrent.futures
import hashlib
import logging
import sys
from unittest import mock
import fixtures
import os_service_types
import testtools
import openstack
from openstack import exceptions
from openstack.tests.unit import base
from openstack import utils
class TestTinyDAG(base.TestCase):
    test_graph = {'a': ['b', 'd', 'f'], 'b': ['c', 'd'], 'c': ['d'], 'd': ['e'], 'e': [], 'f': ['e'], 'g': ['e']}

    def _verify_order(self, test_graph, test_list):
        for k, v in test_graph.items():
            for dep in v:
                self.assertTrue(test_list.index(k) < test_list.index(dep))

    def test_from_dict(self):
        sot = utils.TinyDAG()
        sot.from_dict(self.test_graph)

    def test_topological_sort(self):
        sot = utils.TinyDAG()
        sot.from_dict(self.test_graph)
        sorted_list = sot.topological_sort()
        self._verify_order(sot.graph, sorted_list)
        self.assertEqual(len(self.test_graph.keys()), len(sorted_list))

    def test_walk(self):
        sot = utils.TinyDAG()
        sot.from_dict(self.test_graph)
        sorted_list = []
        for node in sot.walk():
            sorted_list.append(node)
            sot.node_done(node)
        self._verify_order(sot.graph, sorted_list)
        self.assertEqual(len(self.test_graph.keys()), len(sorted_list))

    def test_walk_parallel(self):
        sot = utils.TinyDAG()
        sot.from_dict(self.test_graph)
        sorted_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            for node in sot.walk(timeout=1):
                executor.submit(test_walker_fn, sot, node, sorted_list)
        self._verify_order(sot.graph, sorted_list)
        print(sorted_list)
        self.assertEqual(len(self.test_graph.keys()), len(sorted_list))

    def test_walk_raise(self):
        sot = utils.TinyDAG()
        sot.from_dict(self.test_graph)
        bad_node = 'f'
        with testtools.ExpectedException(exceptions.SDKException):
            for node in sot.walk(timeout=1):
                if node != bad_node:
                    sot.node_done(node)

    def test_add_node_after_edge(self):
        sot = utils.TinyDAG()
        sot.add_node('a')
        sot.add_edge('a', 'b')
        sot.add_node('a')
        self.assertEqual(sot._graph['a'], set('b'))