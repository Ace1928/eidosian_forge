from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildexecutionAdminV1alphaCreateInstanceRequest(_messages.Message):
    """The request used for `CreateInstance`.

  Fields:
    instance: Required. Specifies the instance to create. The name in the
      instance, if specified in the instance, is ignored.
    instanceId: Required. ID of the created instance. A valid `instance_id`
      must: be 6-50 characters long, contain only lowercase letters, digits,
      hyphens and underscores, start with a lowercase letter, and end with a
      lowercase letter or a digit.
    parent: Required. Resource name of the project containing the instance.
      Format: `projects/[PROJECT_ID]`.
  """
    instance = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaInstance', 1)
    instanceId = _messages.StringField(2)
    parent = _messages.StringField(3)