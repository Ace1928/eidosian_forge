from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def CreateFlexTemplateRequest(self, args):
    """Create a Flex Template request for the Pipeline workload.

    Args:
      args: Any, list of args needed to create a Pipeline.

    Returns:
      Flex Template request.
    """
    location = args.region
    project_id = properties.VALUES.core.project.Get(required=True)
    params_list = self.ConvertDictArguments(args.parameters, self.messages.GoogleCloudDatapipelinesV1LaunchFlexTemplateParameter.ParametersValue)
    transform_mapping_list = self.ConvertDictArguments(args.transform_name_mappings, self.messages.GoogleCloudDatapipelinesV1LaunchFlexTemplateParameter.TransformNameMappingsValue)
    transform_name_mappings = None
    if transform_mapping_list:
        transform_name_mappings = self.messages.GoogleCloudDatapipelinesV1LaunchFlexTemplateParameter.TransformNameMappingsValue(additionalProperties=transform_mapping_list)
    ip_private = self.messages.GoogleCloudDatapipelinesV1FlexTemplateRuntimeEnvironment.IpConfigurationValueValuesEnum.WORKER_IP_PRIVATE
    ip_configuration = ip_private if args.disable_public_ips else None
    user_labels_list = self.ConvertDictArguments(args.additional_user_labels, self.messages.GoogleCloudDatapipelinesV1FlexTemplateRuntimeEnvironment.AdditionalUserLabelsValue)
    additional_user_labels = None
    if user_labels_list:
        additional_user_labels = self.messages.GoogleCloudDatapipelinesV1FlexTemplateRuntimeEnvironment.AdditionalUserLabelsValue(additionalProperties=user_labels_list)
    flexrs_goal = None
    if args.flexrs_goal:
        if args.flexrs_goal == 'SPEED_OPTIMIZED':
            flexrs_goal = self.messages.GoogleCloudDatapipelinesV1FlexTemplateRuntimeEnvironment.FlexrsGoalValueValuesEnum.FLEXRS_SPEED_OPTIMIZED
        elif args.flexrs_goal == 'COST_OPTIMIZED':
            flexrs_goal = self.messages.GoogleCloudDatapipelinesV1FlexTemplateRuntimeEnvironment.FlexrsGoalValueValuesEnum.FLEXRS_COST_OPTIMIZED
    launch_parameter = self.messages.GoogleCloudDatapipelinesV1LaunchFlexTemplateParameter(containerSpecGcsPath=args.template_file_gcs_location, environment=self.messages.GoogleCloudDatapipelinesV1FlexTemplateRuntimeEnvironment(serviceAccountEmail=args.dataflow_service_account_email, maxWorkers=args.max_workers, numWorkers=args.num_workers, network=args.network, subnetwork=args.subnetwork, machineType=args.worker_machine_type, tempLocation=args.temp_location, kmsKeyName=args.dataflow_kms_key, ipConfiguration=ip_configuration, workerRegion=args.worker_region, workerZone=args.worker_zone, enableStreamingEngine=args.enable_streaming_engine, flexrsGoal=flexrs_goal, additionalExperiments=args.additional_experiments if args.additional_experiments else [], additionalUserLabels=additional_user_labels), update=args.update, parameters=self.messages.GoogleCloudDatapipelinesV1LaunchFlexTemplateParameter.ParametersValue(additionalProperties=params_list) if params_list else None, transformNameMappings=transform_name_mappings)
    return self.messages.GoogleCloudDatapipelinesV1LaunchFlexTemplateRequest(location=location, projectId=project_id, launchParameter=launch_parameter)