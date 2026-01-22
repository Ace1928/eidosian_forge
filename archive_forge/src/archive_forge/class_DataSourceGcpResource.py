from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataSourceGcpResource(_messages.Message):
    """DataSourceGcpResource is used for protected resources that are GCP
  Resources. This name is easeier to understand than GcpResourceDataSource or
  GcpDataSourceResource

  Fields:
    computeInstanceDatasourceProperties: ComputeInstanceDataSourceProperties
      has a subset of Compute Instance properties that are useful at the
      Datasource level.
    gcpResourcename: Output only. Full resource pathname URL of the source GCP
      resource.
    location: Location of the resource: //"global"/"unspecified".
    type: The type of the GCP resource. Use the Unified Resource Type, eg.
      compute.googleapis.com/Instance.
  """
    computeInstanceDatasourceProperties = _messages.MessageField('ComputeInstanceDataSourceProperties', 1)
    gcpResourcename = _messages.StringField(2)
    location = _messages.StringField(3)
    type = _messages.StringField(4)