from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringServicesCreateRequest(_messages.Message):
    """A MonitoringServicesCreateRequest object.

  Fields:
    parent: Required. Resource name
      (https://cloud.google.com/monitoring/api/v3#project_name) of the parent
      Metrics Scope. The format is: projects/[PROJECT_ID_OR_NUMBER]
    service: A Service resource to be passed as the request body.
    serviceId: Optional. The Service id to use for this Service. If omitted,
      an id will be generated instead. Must match the pattern [a-z0-9\\-]+
  """
    parent = _messages.StringField(1, required=True)
    service = _messages.MessageField('Service', 2)
    serviceId = _messages.StringField(3)