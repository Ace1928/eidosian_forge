from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BasicScaling(_messages.Message):
    """A service with basic scaling will create an instance when the
  application receives a request. The instance will be turned down when the
  app becomes idle. Basic scaling is ideal for work that is intermittent or
  driven by user activity.

  Fields:
    idleTimeout: Duration of time after the last request that an instance must
      wait before the instance is shut down.
    maxInstances: Maximum number of instances to create for this version.
  """
    idleTimeout = _messages.StringField(1)
    maxInstances = _messages.IntegerField(2, variant=_messages.Variant.INT32)