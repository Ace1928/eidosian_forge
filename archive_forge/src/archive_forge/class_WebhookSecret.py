from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WebhookSecret(_messages.Message):
    """Webhook secret referenceable within a WorkflowTrigger.

  Fields:
    id: identification to secret Resource.
    secretVersion: Output only. Secret Manager version.
  """
    id = _messages.StringField(1)
    secretVersion = _messages.StringField(2)