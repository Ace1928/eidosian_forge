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
def _get_output_args(node, stream_name_map):
    if node.name != output.__name__:
        raise ValueError('Unsupported output node: {}'.format(node))
    args = []
    if len(node.incoming_edges) == 0:
        raise ValueError('Output node {} has no mapped streams'.format(node))
    for edge in node.incoming_edges:
        stream_name = _format_input_stream_name(stream_name_map, edge, is_final_arg=True)
        if stream_name != '0' or len(node.incoming_edges) > 1:
            args += ['-map', stream_name]
    kwargs = copy.copy(node.kwargs)
    filename = kwargs.pop('filename')
    if 'format' in kwargs:
        args += ['-f', kwargs.pop('format')]
    if 'video_bitrate' in kwargs:
        args += ['-b:v', str(kwargs.pop('video_bitrate'))]
    if 'audio_bitrate' in kwargs:
        args += ['-b:a', str(kwargs.pop('audio_bitrate'))]
    if 'video_size' in kwargs:
        video_size = kwargs.pop('video_size')
        if not isinstance(video_size, basestring) and isinstance(video_size, collections.Iterable):
            video_size = '{}x{}'.format(video_size[0], video_size[1])
        args += ['-video_size', video_size]
    args += convert_kwargs_to_cmd_line_args(kwargs)
    args += [filename]
    return args