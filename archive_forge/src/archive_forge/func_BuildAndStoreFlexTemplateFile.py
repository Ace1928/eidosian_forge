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
def BuildAndStoreFlexTemplateFile(template_file_gcs_location, image, template_metadata_json, sdk_language, print_only, template_args=None, image_repository_username_secret_id=None, image_repository_password_secret_id=None, image_repository_cert_path=None):
    """Builds container spec and stores it in the flex template file in GCS.

    Args:
      template_file_gcs_location: GCS location to store the template file.
      image: Path to the container image.
      template_metadata_json: Template metadata in json format.
      sdk_language: SDK language of the flex template.
      print_only: Only prints the container spec and skips write to GCS.
      template_args: Default runtime parameters specified by template authors.
      image_repository_username_secret_id: Secret manager secret id for username
        to authenticate to private registry.
      image_repository_password_secret_id: Secret manager secret id for password
        to authenticate to private registry.
      image_repository_cert_path: The full URL to self-signed certificate of
        private registry in Cloud Storage.

    Returns:
      Container spec json if print_only is set. A success message with template
      file GCS path and container spec otherewise.
    """
    template_metadata = None
    if template_metadata_json:
        template_metadata = Templates._BuildTemplateMetadata(template_metadata_json)
    sdk_info = Templates._BuildSDKInfo(sdk_language)
    default_environment = None
    if template_args:
        user_labels_list = Templates.__ConvertDictArguments(template_args.additional_user_labels, Templates.FLEX_TEMPLATE_USER_LABELS_VALUE)
        ip_private = Templates.IP_CONFIGURATION_ENUM_VALUE.WORKER_IP_PRIVATE
        ip_configuration = ip_private if template_args.disable_public_ips else None
        enable_streaming_engine = True if template_args.enable_streaming_engine else None
        default_environment = Templates.FLEX_TEMPLATE_ENVIRONMENT(serviceAccountEmail=template_args.service_account_email, maxWorkers=template_args.max_workers, numWorkers=template_args.num_workers, network=template_args.network, subnetwork=template_args.subnetwork, machineType=template_args.worker_machine_type, tempLocation=template_args.temp_location if template_args.temp_location else template_args.staging_location, stagingLocation=template_args.staging_location, kmsKeyName=template_args.kms_key_name, ipConfiguration=ip_configuration, workerRegion=template_args.worker_region, workerZone=template_args.worker_zone, enableStreamingEngine=enable_streaming_engine, additionalExperiments=template_args.additional_experiments if template_args.additional_experiments else [], additionalUserLabels=Templates.FLEX_TEMPLATE_USER_LABELS_VALUE(additionalProperties=user_labels_list) if user_labels_list else None)
    container_spec = Templates.CONTAINER_SPEC(image=image, metadata=template_metadata, sdkInfo=sdk_info, defaultEnvironment=default_environment, imageRepositoryUsernameSecretId=image_repository_username_secret_id, imageRepositoryPasswordSecretId=image_repository_password_secret_id, imageRepositoryCertPath=image_repository_cert_path)
    container_spec_json = encoding.MessageToJson(container_spec)
    container_spec_pretty_json = json.dumps(json.loads(container_spec_json), sort_keys=True, indent=4, separators=(',', ': '))
    if print_only:
        return container_spec_pretty_json
    try:
        Templates._StoreFlexTemplateFile(template_file_gcs_location, container_spec_pretty_json)
        log.status.Print('Successfully saved container spec in flex template file.\nTemplate File GCS Location: {}\nContainer Spec:\n\n{}'.format(template_file_gcs_location, container_spec_pretty_json))
    except apitools_exceptions.HttpError as error:
        raise exceptions.HttpException(error)