from __future__ import unicode_literals
from .dag import get_outgoing_edges, topo_sort
from ._utils import basestring, convert_kwargs_to_cmd_line_args
from builtins import str
from functools import reduce
import collections
import copy
import operator
import subprocess
from ._ffmpeg import input, output
from .nodes import (
def _allocate_filter_stream_names(filter_nodes, outgoing_edge_maps, stream_name_map):
    stream_count = 0
    for upstream_node in filter_nodes:
        outgoing_edge_map = outgoing_edge_maps[upstream_node]
        for upstream_label, downstreams in sorted(outgoing_edge_map.items()):
            if len(downstreams) > 1:
                raise ValueError('Encountered {} with multiple outgoing edges with same upstream label {!r}; a `split` filter is probably required'.format(upstream_node, upstream_label))
            stream_name_map[upstream_node, upstream_label] = 's{}'.format(stream_count)
            stream_count += 1