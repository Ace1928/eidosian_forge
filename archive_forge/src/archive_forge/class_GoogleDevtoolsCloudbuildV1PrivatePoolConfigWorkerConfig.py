from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsCloudbuildV1PrivatePoolConfigWorkerConfig(_messages.Message):
    """Defines the configuration to be used for creating workers in the pool.

  Fields:
    machineType: Machine type of the workers in the pool. This field does not
      currently support mutations.
  """
    machineType = _messages.StringField(1)