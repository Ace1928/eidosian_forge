from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ServiceSpec(_messages.Message):
    """Specification that applies to a Service resource. Valid only for entries
  with the `SERVICE` type.

  Fields:
    cloudBigtableInstanceSpec: Specification that applies to Instance entries
      of `CLOUD_BIGTABLE` system.
  """
    cloudBigtableInstanceSpec = _messages.MessageField('GoogleCloudDatacatalogV1CloudBigtableInstanceSpec', 1)