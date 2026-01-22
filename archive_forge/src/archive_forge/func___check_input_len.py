from __future__ import unicode_literals
from past.builtins import basestring
from .dag import KwargReprNode
from ._utils import escape_chars, get_hash_int
from builtins import object
import os
@classmethod
def __check_input_len(cls, stream_map, min_inputs, max_inputs):
    if min_inputs is not None and len(stream_map) < min_inputs:
        raise ValueError('Expected at least {} input stream(s); got {}'.format(min_inputs, len(stream_map)))
    elif max_inputs is not None and len(stream_map) > max_inputs:
        raise ValueError('Expected at most {} input stream(s); got {}'.format(max_inputs, len(stream_map)))