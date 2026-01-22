from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsInstancesDetachLunRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsInstancesDetachLunRequest object.

  Fields:
    detachLunRequest: A DetachLunRequest resource to be passed as the request
      body.
    instance: Required. Name of the instance.
  """
    detachLunRequest = _messages.MessageField('DetachLunRequest', 1)
    instance = _messages.StringField(2, required=True)