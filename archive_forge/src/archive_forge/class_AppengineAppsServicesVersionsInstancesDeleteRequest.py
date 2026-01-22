from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsServicesVersionsInstancesDeleteRequest(_messages.Message):
    """A AppengineAppsServicesVersionsInstancesDeleteRequest object.

  Fields:
    name: Name of the resource requested. Example:
      apps/myapp/services/default/versions/v1/instances/instance-1.
  """
    name = _messages.StringField(1, required=True)