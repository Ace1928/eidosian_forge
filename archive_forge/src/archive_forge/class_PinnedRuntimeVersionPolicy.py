from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PinnedRuntimeVersionPolicy(_messages.Message):
    """The function is pinned to a specific runtime version and it will not
  receive security patches, even after redeploying.

  Fields:
    runtimeVersion: The runtime version this function is pinned to. This
      version will be used every time this function is deployed.
  """
    runtimeVersion = _messages.StringField(1)