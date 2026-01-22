from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BatchMigrateResourcesRequest(_messages.Message):
    """Request message for MigrationService.BatchMigrateResources.

  Fields:
    migrateResourceRequests: Required. The request messages specifying the
      resources to migrate. They must be in the same location as the
      destination. Up to 50 resources can be migrated in one batch.
  """
    migrateResourceRequests = _messages.MessageField('GoogleCloudAiplatformV1beta1MigrateResourceRequest', 1, repeated=True)