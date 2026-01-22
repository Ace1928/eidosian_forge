from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1TraceConfigOverride(_messages.Message):
    """A representation of a configuration override.

  Fields:
    apiProxy: ID of the API proxy that will have its trace configuration
      overridden.
    name: ID of the trace configuration override specified as a system-
      generated UUID.
    samplingConfig: Trace configuration to override.
  """
    apiProxy = _messages.StringField(1)
    name = _messages.StringField(2)
    samplingConfig = _messages.MessageField('GoogleCloudApigeeV1TraceSamplingConfig', 3)