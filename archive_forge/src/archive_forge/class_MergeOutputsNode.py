from __future__ import unicode_literals
from past.builtins import basestring
from .dag import KwargReprNode
from ._utils import escape_chars, get_hash_int
from builtins import object
import os
class MergeOutputsNode(Node):

    def __init__(self, streams, name):
        super(MergeOutputsNode, self).__init__(stream_spec=streams, name=name, incoming_stream_types={OutputStream}, outgoing_stream_type=OutputStream, min_inputs=1, max_inputs=None)