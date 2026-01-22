from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthCheckLogConfig(_messages.Message):
    """Configuration of logging on a health check. If logging is enabled, logs
  will be exported to Stackdriver.

  Fields:
    enable: Indicates whether or not to export logs. This is false by default,
      which means no health check logging will be done.
  """
    enable = _messages.BooleanField(1)