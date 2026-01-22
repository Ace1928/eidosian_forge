from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalChannelWithScore(_messages.Message):
    """The channel with score.

  Fields:
    frequencyRange: The frequency range of the channel.
    score: The channel score, normalized to be in the range [0,100].
  """
    frequencyRange = _messages.MessageField('SasPortalFrequencyRange', 1)
    score = _messages.FloatField(2)