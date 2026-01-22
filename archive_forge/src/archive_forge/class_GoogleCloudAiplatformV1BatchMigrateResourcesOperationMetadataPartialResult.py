from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1BatchMigrateResourcesOperationMetadataPartialResult(_messages.Message):
    """Represents a partial result in batch migration operation for one
  MigrateResourceRequest.

  Fields:
    dataset: Migrated dataset resource name.
    error: The error result of the migration request in case of failure.
    model: Migrated model resource name.
    request: It's the same as the value in
      MigrateResourceRequest.migrate_resource_requests.
  """
    dataset = _messages.StringField(1)
    error = _messages.MessageField('GoogleRpcStatus', 2)
    model = _messages.StringField(3)
    request = _messages.MessageField('GoogleCloudAiplatformV1MigrateResourceRequest', 4)