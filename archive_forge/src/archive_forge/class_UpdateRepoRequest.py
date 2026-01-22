from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateRepoRequest(_messages.Message):
    """Request for UpdateRepo.

  Fields:
    repo: The new configuration for the repository.
    updateMask: A FieldMask specifying which fields of the repo to modify.
      Only the fields in the mask will be modified. If no mask is provided,
      this request is no-op.
  """
    repo = _messages.MessageField('Repo', 1)
    updateMask = _messages.StringField(2)