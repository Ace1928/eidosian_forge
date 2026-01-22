from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunappsProjectsLocationsApplicationsGetStatusRequest(_messages.Message):
    """A RunappsProjectsLocationsApplicationsGetStatusRequest object.

  Fields:
    name: Required. Name of the resource.
    readMask: Field mask used for limiting the resources to query status on.
    resources: Optional. Specify which resource to query status for. If not
      provided, all resources status are queried.
  """
    name = _messages.StringField(1, required=True)
    readMask = _messages.StringField(2)
    resources = _messages.StringField(3, repeated=True)