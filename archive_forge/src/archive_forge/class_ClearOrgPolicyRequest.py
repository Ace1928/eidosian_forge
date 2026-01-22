from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClearOrgPolicyRequest(_messages.Message):
    """The request sent to the ClearOrgPolicy method.

  Fields:
    constraint: Name of the `Constraint` of the `Policy` to clear.
    etag: The current version, for concurrency control. Not sending an `etag`
      will cause the `Policy` to be cleared blindly.
  """
    constraint = _messages.StringField(1)
    etag = _messages.BytesField(2)