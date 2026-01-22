from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Telemetry(_messages.Message):
    """Configuration for how to query telemetry on a Service.

  Fields:
    resourceName: The full name of the resource that defines this service.
      Formatted as described in
      https://cloud.google.com/apis/design/resource_names.
  """
    resourceName = _messages.StringField(1)