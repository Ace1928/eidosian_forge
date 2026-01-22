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
def DeleteResources(self, manifests, region, force):
    """Delete Cloud Deploy resources.

    Asynchronously calls the API then iterate the operations
    to check the status.

    Args:
     manifests: [str], the list of parsed resource yaml definitions.
     region: str, location ID.
     force: bool, if true, the delivery pipeline with sub-resources will be
       deleted and its sub-resources will also be deleted.
    """
    resource_dict = manifest_util.ParseDeployConfig(self.messages, manifests, region)
    msg_template = 'Deleted Cloud Deploy resource: {}.'
    targets = resource_dict[manifest_util.TARGET_KIND_V1BETA1]
    if targets:
        operation_dict = {}
        for resource in targets:
            operation_dict[resource.name] = target_util.DeleteTarget(resource.name)
        self.operation_client.CheckOperationStatus(operation_dict, msg_template)
    custom_target_types = resource_dict[manifest_util.CUSTOM_TARGET_TYPE_KIND]
    if custom_target_types:
        operation_dict = {}
        for resource in custom_target_types:
            operation_dict[resource.name] = custom_target_type_util.DeleteCustomTargetType(resource.name)
        self.operation_client.CheckOperationStatus(operation_dict, msg_template)
    automations = resource_dict[manifest_util.AUTOMATION_KIND]
    operation_dict = {}
    for resource in automations:
        operation_dict[resource.name] = automation_util.DeleteAutomation(resource.name)
        self.operation_client.CheckOperationStatus(operation_dict, msg_template)
    pipelines = resource_dict[manifest_util.DELIVERY_PIPELINE_KIND_V1BETA1]
    if pipelines:
        operation_dict = {}
        for resource in pipelines:
            operation_dict[resource.name] = self.DeleteDeliveryPipeline(resource, force)
        self.operation_client.CheckOperationStatus(operation_dict, msg_template)
    deploy_policies = resource_dict[manifest_util.DEPLOY_POLICY_KIND]
    if deploy_policies:
        operation_dict = {}
        for resource in deploy_policies:
            operation_dict[resource.name] = deploy_policy_util.DeleteDeployPolicy(resource.name)
        self.operation_client.CheckOperationStatus(operation_dict, msg_template)