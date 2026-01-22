from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OverrideInlineSource(_messages.Message):
    """Import data embedded in the request message

  Fields:
    overrides: The overrides to create. Each override must have a value for
      'metric' and 'unit', to specify which metric and which limit the
      override should be applied to.
  """
    overrides = _messages.MessageField('QuotaOverride', 1, repeated=True)