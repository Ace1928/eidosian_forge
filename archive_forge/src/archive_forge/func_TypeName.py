import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def TypeName(self):
    """Returns the protobuf type name of the inner message."""
    return self.type_url.split('/')[-1]