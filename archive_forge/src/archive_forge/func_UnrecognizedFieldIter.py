import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def UnrecognizedFieldIter(message, _edges=()):
    """Yields the locations of unrecognized fields within "message".

    If a sub-message is found to have unrecognized fields, that sub-message
    will not be searched any further. We prune the search of the sub-message
    because we assume it is malformed and further checks will not yield
    productive errors.

    Args:
      message: The Message instance to search.
      _edges: Internal arg for passing state.

    Yields:
      (edges_to_message, field_names):
        edges_to_message: List[ProtoEdge], The edges (relative to "message")
            describing the path to the sub-message where the unrecognized
            fields were found.
        field_names: List[Str], The names of the field(s) that were
            unrecognized in the sub-message.
    """
    if not isinstance(message, messages.Message):
        return
    field_names = message.all_unrecognized_fields()
    if field_names:
        yield (_edges, field_names)
        return
    for field in message.all_fields():
        value = message.get_assigned_value(field.name)
        if field.repeated:
            for i, item in enumerate(value):
                repeated_edge = ProtoEdge(EdgeType.REPEATED, field.name, i)
                iter_ = UnrecognizedFieldIter(item, _edges + (repeated_edge,))
                for e, y in iter_:
                    yield (e, y)
        elif _IsMap(message, field):
            for key, item in _MapItems(message, field):
                map_edge = ProtoEdge(EdgeType.MAP, field.name, key)
                iter_ = UnrecognizedFieldIter(item, _edges + (map_edge,))
                for e, y in iter_:
                    yield (e, y)
        else:
            scalar_edge = ProtoEdge(EdgeType.SCALAR, field.name, None)
            iter_ = UnrecognizedFieldIter(value, _edges + (scalar_edge,))
            for e, y in iter_:
                yield (e, y)