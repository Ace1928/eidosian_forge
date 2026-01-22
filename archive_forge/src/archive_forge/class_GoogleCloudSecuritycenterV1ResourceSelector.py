from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV1ResourceSelector(_messages.Message):
    """Resource for selecting resource type.

  Fields:
    resourceTypes: The resource types to run the detector on.
  """
    resourceTypes = _messages.StringField(1, repeated=True)