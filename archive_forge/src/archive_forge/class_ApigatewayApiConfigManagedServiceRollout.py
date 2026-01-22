from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayApiConfigManagedServiceRollout(_messages.Message):
    """Rollout for a Managed Service ( https://cloud.google.com/service-
  infrastructure/docs/glossary#managed).

  Fields:
    rolloutId: Optional. The Rollout ID for the Managed Service. See
      https://cloud.google.com/service-infrastructure/docs/rollout for more
      info.
  """
    rolloutId = _messages.StringField(1)