from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GlobalOrganizationSetPolicyRequest(_messages.Message):
    """A GlobalOrganizationSetPolicyRequest object.

  Fields:
    bindings: Flatten Policy to create a backward compatible wire-format.
      Deprecated. Use 'policy' to specify bindings.
    etag: Flatten Policy to create a backward compatible wire-format.
      Deprecated. Use 'policy' to specify the etag.
    policy: REQUIRED: The complete policy to be applied to the 'resource'. The
      size of the policy is limited to a few 10s of KB. An empty policy is in
      general a valid policy but certain services (like Projects) might reject
      them.
  """
    bindings = _messages.MessageField('Binding', 1, repeated=True)
    etag = _messages.BytesField(2)
    policy = _messages.MessageField('Policy', 3)