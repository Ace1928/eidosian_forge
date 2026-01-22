from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsPatchJobsCancelRequest(_messages.Message):
    """A OsconfigProjectsPatchJobsCancelRequest object.

  Fields:
    cancelPatchJobRequest: A CancelPatchJobRequest resource to be passed as
      the request body.
    name: Required. Name of the patch in the form `projects/*/patchJobs/*`
  """
    cancelPatchJobRequest = _messages.MessageField('CancelPatchJobRequest', 1)
    name = _messages.StringField(2, required=True)