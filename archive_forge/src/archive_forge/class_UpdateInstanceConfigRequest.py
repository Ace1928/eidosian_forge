from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateInstanceConfigRequest(_messages.Message):
    """The request for UpdateInstanceConfigRequest.

  Fields:
    instanceConfig: Required. The user instance config to update, which must
      always include the instance config name. Otherwise, only fields
      mentioned in update_mask need be included. To prevent conflicts of
      concurrent updates, etag can be used.
    updateMask: Required. A mask specifying which fields in InstanceConfig
      should be updated. The field mask must always be specified; this
      prevents any future fields in InstanceConfig from being erased
      accidentally by clients that do not know about them. Only display_name
      and labels can be updated.
    validateOnly: An option to validate, but not actually execute, a request,
      and provide the same response.
  """
    instanceConfig = _messages.MessageField('InstanceConfig', 1)
    updateMask = _messages.StringField(2)
    validateOnly = _messages.BooleanField(3)