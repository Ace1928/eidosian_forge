from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsInstancesReimageRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsInstancesReimageRequest object.

  Fields:
    name: Required. The `name` field is used to identify the instance. Format:
      projects/{project}/locations/{location}/instances/{instance}
    reimageInstanceRequest: A ReimageInstanceRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    reimageInstanceRequest = _messages.MessageField('ReimageInstanceRequest', 2)