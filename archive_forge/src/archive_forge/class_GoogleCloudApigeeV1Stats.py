from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Stats(_messages.Message):
    """Encapsulates a `stats` response.

  Fields:
    environments: List of query results on the environment level.
    hosts: List of query results grouped by host.
    metaData: Metadata information.
  """
    environments = _messages.MessageField('GoogleCloudApigeeV1StatsEnvironmentStats', 1, repeated=True)
    hosts = _messages.MessageField('GoogleCloudApigeeV1StatsHostStats', 2, repeated=True)
    metaData = _messages.MessageField('GoogleCloudApigeeV1Metadata', 3)