import collections
import contextlib
import itertools
import threading
from automaton import runners
from concurrent import futures
import fasteners
import functools
import networkx as nx
from oslo_utils import excutils
from oslo_utils import strutils
from oslo_utils import timeutils
from taskflow.engines.action_engine import builder
from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import executor
from taskflow.engines.action_engine import process_executor
from taskflow.engines.action_engine import runtime
from taskflow.engines import base
from taskflow import exceptions as exc
from taskflow import logging
from taskflow import states
from taskflow import storage
from taskflow.types import failure
from taskflow.utils import misc
def _ensure_storage(self):
    """Ensure all contained atoms exist in the storage unit."""
    self.storage.ensure_atoms(self._runtime.iterate_nodes(compiler.ATOMS))
    for atom in self._runtime.iterate_nodes(compiler.ATOMS):
        if atom.inject:
            self.storage.inject_atom_args(atom.name, atom.inject, transient=self._inject_transient)