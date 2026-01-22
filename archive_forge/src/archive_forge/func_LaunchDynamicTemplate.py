from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import shutil
import textwrap
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.command_lib.builds import submit_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
@staticmethod
def LaunchDynamicTemplate(template_args=None):
    """Calls the Dataflow Templates.LaunchTemplate method on a dynamic template.

    Args:
      template_args: Arguments to create template. gcs_location must point to a
        Json serialized DynamicTemplateFileSpec.

    Returns:
      (LaunchTemplateResponse)
    """
    params_list = []
    parameters = template_args.parameters
    for k, v in six.iteritems(parameters) if parameters else {}:
        params_list.append(Templates.LAUNCH_TEMPLATE_PARAMETERS_VALUE.AdditionalProperty(key=k, value=v))
    region_id = template_args.region_id or DATAFLOW_API_DEFAULT_REGION
    ip_configuration_enum = GetMessagesModule().RuntimeEnvironment.IpConfigurationValueValuesEnum
    ip_private = ip_configuration_enum.WORKER_IP_PRIVATE
    ip_configuration = ip_private if template_args.disable_public_ips else None
    body = Templates.LAUNCH_TEMPLATE_PARAMETERS(environment=GetMessagesModule().RuntimeEnvironment(serviceAccountEmail=template_args.service_account_email, zone=template_args.zone, maxWorkers=template_args.max_workers, numWorkers=template_args.num_workers, network=template_args.network, subnetwork=template_args.subnetwork, machineType=template_args.worker_machine_type, tempLocation=template_args.staging_location, kmsKeyName=template_args.kms_key_name, ipConfiguration=ip_configuration, workerRegion=template_args.worker_region, workerZone=template_args.worker_zone), jobName=template_args.job_name, parameters=Templates.LAUNCH_TEMPLATE_PARAMETERS_VALUE(additionalProperties=params_list) if parameters else None, update=False)
    request = GetMessagesModule().DataflowProjectsLocationsTemplatesLaunchRequest(dynamicTemplate_gcsPath=template_args.gcs_location, dynamicTemplate_stagingLocation=template_args.staging_location, location=region_id, launchTemplateParameters=body, projectId=template_args.project_id or GetProject(), validateOnly=False)
    Templates.ModifyDynamicTemplatesLaunchRequest(request)
    try:
        return Templates.GetService().Launch(request)
    except apitools_exceptions.HttpError as error:
        raise exceptions.HttpException(error)