from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeRegionHealthCheckServicesGetRequest(_messages.Message):
    """A ComputeRegionHealthCheckServicesGetRequest object.

  Fields:
    healthCheckService: Name of the HealthCheckService to update. The name
      must be 1-63 characters long, and comply with RFC1035.
    project: Project ID for this request.
    region: Name of the region scoping this request.
  """
    healthCheckService = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    region = _messages.StringField(3, required=True)