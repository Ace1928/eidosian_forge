from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnthosSseConfig(_messages.Message):
    """Configuration options for the Anthos SSE bundle.

  Fields:
    enabled: Whether the Anthos SSE bundle is enabled on the KrmApiHost.
  """
    enabled = _messages.BooleanField(1)