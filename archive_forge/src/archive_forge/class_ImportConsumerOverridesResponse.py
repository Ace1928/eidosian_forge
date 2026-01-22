from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportConsumerOverridesResponse(_messages.Message):
    """Response message for ImportConsumerOverrides

  Fields:
    overrides: The overrides that were created from the imported data.
  """
    overrides = _messages.MessageField('QuotaOverride', 1, repeated=True)