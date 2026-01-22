from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IAMPolicy(_messages.Message):
    """IAMPolicy encapsulates the IAM policy name, definition and status of
  policy fetching.

  Fields:
    policy: Policy definition if IAM policy fetching is successful, otherwise
      empty.
    status: Status of iam policy fetching.
  """
    policy = _messages.MessageField('Policy', 1)
    status = _messages.MessageField('Status', 2)