from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuotaLimitOverride(_messages.Message):
    """Specifies a custom quota limit that is applied for this consumer
  project. This overrides the default value in google.api.QuotaLimit.

  Fields:
    limit: The new limit for this project. May be -1 (unlimited), 0 (block),
      or any positive integer.
    unlimited: Indicates the override is to provide unlimited quota.  If true,
      any value set for limit will be ignored. DEPRECATED. Use a limit value
      of -1 instead.
  """
    limit = _messages.IntegerField(1)
    unlimited = _messages.BooleanField(2)