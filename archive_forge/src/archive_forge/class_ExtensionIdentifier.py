from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
class ExtensionIdentifier(object):
    __slots__ = ('full_name', 'number', 'field_type', 'wire_tag', 'is_repeated', 'default', 'containing_cls', 'composite_cls', 'message_name')

    def __init__(self, full_name, number, field_type, wire_tag, is_repeated, default):
        self.full_name = full_name
        self.number = number
        self.field_type = field_type
        self.wire_tag = wire_tag
        self.is_repeated = is_repeated
        self.default = default