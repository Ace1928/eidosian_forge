from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V1Beta1ImportProducerOverridesResponse(_messages.Message):
    """Response message for ImportProducerOverrides

  Fields:
    overrides: The overrides that were created from the imported data.
  """
    overrides = _messages.MessageField('V1Beta1QuotaOverride', 1, repeated=True)