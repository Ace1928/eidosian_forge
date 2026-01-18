from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import encoding
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.storage import metadata_util
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.api_lib.storage.gcs_json import metadata_field_converters
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import gzip_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.resources import gcs_resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import properties
def update_bucket_metadata_from_request_config(bucket_metadata, request_config):
    """Sets Apitools Bucket fields based on values in request_config."""
    resource_args = getattr(request_config, 'resource_args', None)
    if not resource_args:
        return
    if resource_args.enable_autoclass is not None or resource_args.autoclass_terminal_storage_class is not None:
        bucket_metadata.autoclass = metadata_field_converters.process_autoclass(resource_args.enable_autoclass, resource_args.autoclass_terminal_storage_class)
    if resource_args.enable_hierarchical_namespace is not None:
        bucket_metadata.hierarchicalNamespace = metadata_field_converters.process_hierarchical_namespace(resource_args.enable_hierarchical_namespace)
    if resource_args.cors_file_path is not None:
        bucket_metadata.cors = metadata_field_converters.process_cors(resource_args.cors_file_path)
    if resource_args.default_encryption_key is not None:
        bucket_metadata.encryption = metadata_field_converters.process_default_encryption_key(resource_args.default_encryption_key)
    if resource_args.default_event_based_hold is not None:
        bucket_metadata.defaultEventBasedHold = resource_args.default_event_based_hold
    if resource_args.default_storage_class is not None:
        bucket_metadata.storageClass = metadata_field_converters.process_default_storage_class(resource_args.default_storage_class)
    if resource_args.lifecycle_file_path is not None:
        bucket_metadata.lifecycle = metadata_field_converters.process_lifecycle(resource_args.lifecycle_file_path)
    if resource_args.location is not None:
        bucket_metadata.location = resource_args.location
    if resource_args.log_bucket is not None or resource_args.log_object_prefix is not None:
        bucket_metadata.logging = metadata_field_converters.process_log_config(bucket_metadata.name, resource_args.log_bucket, resource_args.log_object_prefix)
    if resource_args.placement is not None:
        bucket_metadata.customPlacementConfig = metadata_field_converters.process_placement_config(resource_args.placement)
    if resource_args.public_access_prevention is not None or resource_args.uniform_bucket_level_access is not None:
        bucket_metadata.iamConfiguration = metadata_field_converters.process_bucket_iam_configuration(bucket_metadata.iamConfiguration, resource_args.public_access_prevention, resource_args.uniform_bucket_level_access)
    if resource_args.recovery_point_objective is not None:
        bucket_metadata.rpo = resource_args.recovery_point_objective
    if resource_args.requester_pays is not None:
        bucket_metadata.billing = metadata_field_converters.process_requester_pays(bucket_metadata.billing, resource_args.requester_pays)
    if resource_args.retention_period is not None:
        bucket_metadata.retentionPolicy = metadata_field_converters.process_retention_period(resource_args.retention_period)
    if resource_args.soft_delete_duration is not None:
        bucket_metadata.softDeletePolicy = metadata_field_converters.process_soft_delete_duration(resource_args.soft_delete_duration)
    if resource_args.versioning is not None:
        bucket_metadata.versioning = metadata_field_converters.process_versioning(resource_args.versioning)
    if resource_args.web_error_page is not None or resource_args.web_main_page_suffix is not None:
        bucket_metadata.website = metadata_field_converters.process_website(resource_args.web_error_page, resource_args.web_main_page_suffix)
    if resource_args.acl_file_path is not None:
        bucket_metadata.acl = metadata_field_converters.process_acl_file(resource_args.acl_file_path, is_bucket=True)
    bucket_metadata.acl = _get_list_with_added_and_removed_acl_grants(bucket_metadata.acl, resource_args, is_bucket=True, is_default_object_acl=False)
    if resource_args.default_object_acl_file_path is not None:
        bucket_metadata.defaultObjectAcl = metadata_field_converters.process_acl_file(resource_args.default_object_acl_file_path, is_bucket=False)
    bucket_metadata.defaultObjectAcl = _get_list_with_added_and_removed_acl_grants(bucket_metadata.defaultObjectAcl, resource_args, is_bucket=False, is_default_object_acl=True)
    if resource_args.labels_file_path is not None:
        bucket_metadata.labels = metadata_field_converters.process_labels(bucket_metadata.labels, resource_args.labels_file_path)
    bucket_metadata.labels = _get_labels_object_with_added_and_removed_labels(bucket_metadata.labels, resource_args)