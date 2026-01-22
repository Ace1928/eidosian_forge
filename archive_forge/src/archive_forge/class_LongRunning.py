from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LongRunning(_messages.Message):
    """Describes settings to use when generating API methods that use the long-
  running operation pattern. All default values below are from those used in
  the client library generators (e.g.
  [Java](https://github.com/googleapis/gapic-generator-java/blob/04c2faa191a9b
  5a10b92392fe8482279c4404803/src/main/java/com/google/api/generator/gapic/com
  poser/common/RetrySettingsComposer.java)).

  Fields:
    initialPollDelay: Initial delay after which the first poll request will be
      made. Default value: 5 seconds.
    maxPollDelay: Maximum time between two subsequent poll requests. Default
      value: 45 seconds.
    pollDelayMultiplier: Multiplier to gradually increase delay between
      subsequent polls until it reaches max_poll_delay. Default value: 1.5.
    totalPollTimeout: Total polling timeout. Default value: 5 minutes.
  """
    initialPollDelay = _messages.StringField(1)
    maxPollDelay = _messages.StringField(2)
    pollDelayMultiplier = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    totalPollTimeout = _messages.StringField(4)