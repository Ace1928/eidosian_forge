from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StackdriverLoggingConfig(_messages.Message):
    """Configuration options for writing logs to [Stackdriver
  Logging](https://cloud.google.com/logging/docs/).

  Fields:
    samplingRatio: Specifies the fraction of operations to write to
      [Stackdriver Logging](https://cloud.google.com/logging/docs/). This
      field may contain any value between 0.0 and 1.0, inclusive. 0.0 is the
      default and means that no operations are logged.
  """
    samplingRatio = _messages.FloatField(1)