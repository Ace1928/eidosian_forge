from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceTypeValueValuesEnum(_messages.Enum):
    """The `InstanceType` of the current instance.

    Values:
      INSTANCE_TYPE_UNSPECIFIED: Not specified.
      PROVISIONED: Provisioned instances have dedicated resources, standard
        usage limits and support.
      FREE_INSTANCE: Free instances provide no guarantee for dedicated
        resources, [node_count, processing_units] should be 0. They come with
        stricter usage limits and limited support.
    """
    INSTANCE_TYPE_UNSPECIFIED = 0
    PROVISIONED = 1
    FREE_INSTANCE = 2