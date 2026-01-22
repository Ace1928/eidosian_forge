from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FailurePolicy(_messages.Message):
    """Describes the policy in case of function's execution failure. If empty,
  then defaults to ignoring failures (i.e. not retrying them).

  Fields:
    retry: If specified, then the function will be retried in case of a
      failure.
  """
    retry = _messages.MessageField('Retry', 1)