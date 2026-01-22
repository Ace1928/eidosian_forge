from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.command_lib.deploy import automation_util
from googlecloudsdk.command_lib.deploy import custom_target_type_util
from googlecloudsdk.command_lib.deploy import deploy_policy_util
from googlecloudsdk.command_lib.deploy import manifest_util
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.core import log
def DeleteDeliveryPipeline(self, pipeline_config, force):
    """Deletes a delivery pipeline resource.

    Args:
      pipeline_config: apitools.base.protorpclite.messages.Message, delivery
        pipeline message.
      force: if true, the delivery pipeline with sub-resources will be deleted
        and its sub-resources will also be deleted.

    Returns:
      The operation message. It could be none if the resource doesn't exist.
    """
    log.debug('Deleting delivery pipeline: ' + repr(pipeline_config))
    return self._pipeline_service.Delete(self.messages.ClouddeployProjectsLocationsDeliveryPipelinesDeleteRequest(allowMissing=True, name=pipeline_config.name, force=force))