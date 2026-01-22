from __future__ import unicode_literals
from past.builtins import basestring
from .dag import KwargReprNode
from ._utils import escape_chars, get_hash_int
from builtins import object
import os
@classmethod
def __check_input_types(cls, stream_map, incoming_stream_types):
    for stream in list(stream_map.values()):
        if not _is_of_types(stream, incoming_stream_types):
            raise TypeError('Expected incoming stream(s) to be of one of the following types: {}; got {}'.format(_get_types_str(incoming_stream_types), type(stream)))