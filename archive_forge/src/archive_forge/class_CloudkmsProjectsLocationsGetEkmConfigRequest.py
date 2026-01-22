from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsGetEkmConfigRequest(_messages.Message):
    """A CloudkmsProjectsLocationsGetEkmConfigRequest object.

  Fields:
    name: Required. The name of the EkmConfig to get.
  """
    name = _messages.StringField(1, required=True)