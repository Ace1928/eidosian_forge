import logging
from functools import partial
from collections import defaultdict
from os.path import splitext
from .diagrams_base import BaseGraph
def _add_nodes(self, states, container):
    self._add_nested_nodes(states, container, prefix='', default_style='default')