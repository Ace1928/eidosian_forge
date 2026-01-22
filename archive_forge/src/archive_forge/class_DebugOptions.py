from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DebugOptions(_messages.Message):
    """Describes any options that have an effect on the debugging of pipelines.

  Fields:
    dataSampling: Configuration options for sampling elements from a running
      pipeline.
    enableHotKeyLogging: When true, enables the logging of the literal hot key
      to the user's Cloud Logging.
  """
    dataSampling = _messages.MessageField('DataSamplingConfig', 1)
    enableHotKeyLogging = _messages.BooleanField(2)