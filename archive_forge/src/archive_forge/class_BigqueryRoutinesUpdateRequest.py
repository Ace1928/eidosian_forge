from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryRoutinesUpdateRequest(_messages.Message):
    """A BigqueryRoutinesUpdateRequest object.

  Fields:
    datasetId: Required. Dataset ID of the routine to update
    projectId: Required. Project ID of the routine to update
    routine: A Routine resource to be passed as the request body.
    routineId: Required. Routine ID of the routine to update
  """
    datasetId = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)
    routine = _messages.MessageField('Routine', 3)
    routineId = _messages.StringField(4, required=True)