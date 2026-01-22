from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MemcacheProjectsLocationsInstancesUpdateParametersRequest(_messages.Message):
    """A MemcacheProjectsLocationsInstancesUpdateParametersRequest object.

  Fields:
    name: Required. Resource name of the Memcached instance for which the
      parameters should be updated.
    updateParametersRequest: A UpdateParametersRequest resource to be passed
      as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateParametersRequest = _messages.MessageField('UpdateParametersRequest', 2)