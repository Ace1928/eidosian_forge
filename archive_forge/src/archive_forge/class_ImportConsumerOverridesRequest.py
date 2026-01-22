from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportConsumerOverridesRequest(_messages.Message):
    """Request message for ImportConsumerOverrides

  Fields:
    force: Whether to force the creation of the quota overrides. If creating
      an override would cause the effective quota for the consumer to decrease
      by more than 10 percent, the call is rejected, as a safety measure to
      avoid accidentally decreasing quota too quickly. Setting the force
      parameter to true ignores this restriction.
    inlineSource: The import data is specified in the request message itself
  """
    force = _messages.BooleanField(1)
    inlineSource = _messages.MessageField('OverrideInlineSource', 2)