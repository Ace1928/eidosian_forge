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
def _format_output_stream_name(stream_name_map, edge):
    return '[{}]'.format(stream_name_map[edge.upstream_node, edge.upstream_label])