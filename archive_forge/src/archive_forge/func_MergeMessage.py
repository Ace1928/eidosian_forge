import calendar
import collections.abc
from datetime import datetime
from datetime import timedelta
from cloudsdk.google.protobuf.descriptor import FieldDescriptor
def MergeMessage(self, source, destination, replace_message, replace_repeated):
    """Merge all fields specified by this tree from source to destination."""
    _MergeMessage(self._root, source, destination, replace_message, replace_repeated)