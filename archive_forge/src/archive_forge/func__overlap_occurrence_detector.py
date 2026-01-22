import threading
import fasteners
from oslo_utils import excutils
from taskflow import flow
from taskflow import logging
from taskflow import task
from taskflow.types import graph as gr
from taskflow.types import tree as tr
from taskflow.utils import iter_utils
from taskflow.utils import misc
from taskflow.flow import (LINK_INVARIANT, LINK_RETRY)  # noqa
def _overlap_occurrence_detector(to_graph, from_graph):
    """Returns how many nodes in 'from' graph are in 'to' graph (if any)."""
    return iter_utils.count((node for node in from_graph.nodes if node in to_graph))