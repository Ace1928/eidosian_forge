from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReleaseReadyCondition(_messages.Message):
    """ReleaseReadyCondition contains information around the status of the
  Release. If a release is not ready, you cannot create a rollout with the
  release.

  Fields:
    status: True if the Release is in a valid state. Otherwise at least one
      condition in `ReleaseCondition` is in an invalid state. Iterate over
      those conditions and see which condition(s) has status = false to find
      out what is wrong with the Release.
  """
    status = _messages.BooleanField(1)