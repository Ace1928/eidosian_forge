import calendar
import collections.abc
from datetime import datetime
from datetime import timedelta
from cloudsdk.google.protobuf.descriptor import FieldDescriptor
def _AddFieldPaths(node, prefix, field_mask):
    """Adds the field paths descended from node to field_mask."""
    if not node and prefix:
        field_mask.paths.append(prefix)
        return
    for name in sorted(node):
        if prefix:
            child_path = prefix + '.' + name
        else:
            child_path = name
        _AddFieldPaths(node[name], child_path, field_mask)