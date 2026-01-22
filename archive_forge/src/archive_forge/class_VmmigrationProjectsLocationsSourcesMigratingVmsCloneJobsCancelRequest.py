from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmmigrationProjectsLocationsSourcesMigratingVmsCloneJobsCancelRequest(_messages.Message):
    """A VmmigrationProjectsLocationsSourcesMigratingVmsCloneJobsCancelRequest
  object.

  Fields:
    cancelCloneJobRequest: A CancelCloneJobRequest resource to be passed as
      the request body.
    name: Required. The clone job id
  """
    cancelCloneJobRequest = _messages.MessageField('CancelCloneJobRequest', 1)
    name = _messages.StringField(2, required=True)