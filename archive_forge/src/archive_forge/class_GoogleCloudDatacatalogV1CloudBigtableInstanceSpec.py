from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1CloudBigtableInstanceSpec(_messages.Message):
    """Specification that applies to Instance entries that are part of
  `CLOUD_BIGTABLE` system. (user_specified_type)

  Fields:
    cloudBigtableClusterSpecs: The list of clusters for the Instance.
  """
    cloudBigtableClusterSpecs = _messages.MessageField('GoogleCloudDatacatalogV1CloudBigtableInstanceSpecCloudBigtableClusterSpec', 1, repeated=True)