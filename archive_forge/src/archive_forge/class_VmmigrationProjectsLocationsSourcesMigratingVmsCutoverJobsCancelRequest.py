from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmmigrationProjectsLocationsSourcesMigratingVmsCutoverJobsCancelRequest(_messages.Message):
    """A
  VmmigrationProjectsLocationsSourcesMigratingVmsCutoverJobsCancelRequest
  object.

  Fields:
    cancelCutoverJobRequest: A CancelCutoverJobRequest resource to be passed
      as the request body.
    name: Required. The cutover job id
  """
    cancelCutoverJobRequest = _messages.MessageField('CancelCutoverJobRequest', 1)
    name = _messages.StringField(2, required=True)