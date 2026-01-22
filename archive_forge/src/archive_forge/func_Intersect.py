import calendar
import collections.abc
from datetime import datetime
from datetime import timedelta
from cloudsdk.google.protobuf.descriptor import FieldDescriptor
def Intersect(self, mask1, mask2):
    """Intersects mask1 and mask2 into this FieldMask."""
    _CheckFieldMaskMessage(mask1)
    _CheckFieldMaskMessage(mask2)
    tree = _FieldMaskTree(mask1)
    intersection = _FieldMaskTree()
    for path in mask2.paths:
        tree.IntersectPath(path, intersection)
    intersection.ToFieldMask(self)