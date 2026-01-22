from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OperationMetadataV1Alpha1(_messages.Message):
    """Metadata for google.longrunning.Operation.

  Fields:
    endTime: Output only. Time when the operation completed.
    insertTime: Output only. Time when the operation was created.
    method: Output only. Method that initiated the operation e.g.
      google.cloud.vpcaccess.v1alpha1.Connectors.CreateConnector.
    target: Output only. Name of the resource that this operation is acting on
      e.g. projects/my-project/locations/us-central1/connectors/v1.
  """
    endTime = _messages.StringField(1)
    insertTime = _messages.StringField(2)
    method = _messages.StringField(3)
    target = _messages.StringField(4)