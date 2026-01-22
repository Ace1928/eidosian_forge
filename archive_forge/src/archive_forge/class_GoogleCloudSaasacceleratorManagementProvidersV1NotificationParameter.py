from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSaasacceleratorManagementProvidersV1NotificationParameter(_messages.Message):
    """Contains notification related data.

  Fields:
    values: Optional. Array of string values. e.g. instance's replica
      information.
  """
    values = _messages.StringField(1, repeated=True)