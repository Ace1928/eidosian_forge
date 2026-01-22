from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1Config(_messages.Message):
    """A GoogleCloudMlV1Config object.

  Fields:
    tpuServiceAccount: The service account Cloud ML uses to run on TPU node.
  """
    tpuServiceAccount = _messages.StringField(1)