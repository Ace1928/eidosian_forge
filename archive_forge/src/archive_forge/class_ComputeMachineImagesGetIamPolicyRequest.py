from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeMachineImagesGetIamPolicyRequest(_messages.Message):
    """A ComputeMachineImagesGetIamPolicyRequest object.

  Fields:
    optionsRequestedPolicyVersion: Requested IAM Policy version.
    project: Project ID for this request.
    resource: Name or id of the resource for this request.
  """
    optionsRequestedPolicyVersion = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    project = _messages.StringField(2, required=True)
    resource = _messages.StringField(3, required=True)