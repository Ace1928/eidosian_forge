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
def test_walker_fn(graph, node, lst):
    lst.append(node)
    graph.node_done(node)