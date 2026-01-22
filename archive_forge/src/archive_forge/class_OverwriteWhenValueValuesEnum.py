from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OverwriteWhenValueValuesEnum(_messages.Enum):
    """When to overwrite objects that already exist in the sink. If not set,
    overwrite behavior is determined by
    overwrite_objects_already_existing_in_sink.

    Values:
      OVERWRITE_WHEN_UNSPECIFIED: Overwrite behavior is unspecified.
      DIFFERENT: Overwrites destination objects with the source objects, only
        if the objects have the same name but different HTTP ETags or checksum
        values.
      NEVER: Never overwrites a destination object if a source object has the
        same name. In this case, the source object is not transferred.
      ALWAYS: Always overwrite the destination object with the source object,
        even if the HTTP Etags or checksum values are the same.
    """
    OVERWRITE_WHEN_UNSPECIFIED = 0
    DIFFERENT = 1
    NEVER = 2
    ALWAYS = 3