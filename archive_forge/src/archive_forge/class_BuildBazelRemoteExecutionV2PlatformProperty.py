from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildBazelRemoteExecutionV2PlatformProperty(_messages.Message):
    """A single property for the environment. The server is responsible for
  specifying the property `name`s that it accepts. If an unknown `name` is
  provided in the requirements for an Action, the server SHOULD reject the
  execution request. If permitted by the server, the same `name` may occur
  multiple times. The server is also responsible for specifying the
  interpretation of property `value`s. For instance, a property describing how
  much RAM must be available may be interpreted as allowing a worker with 16GB
  to fulfill a request for 8GB, while a property describing the OS environment
  on which the action must be performed may require an exact match with the
  worker's OS. The server MAY use the `value` of one or more properties to
  determine how it sets up the execution environment, such as by making
  specific system files available to the worker. Both names and values are
  typically case-sensitive. Note that the platform is implicitly part of the
  action digest, so even tiny changes in the names or values (like changing
  case) may result in different action cache entries.

  Fields:
    name: The property name.
    value: The property value.
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)