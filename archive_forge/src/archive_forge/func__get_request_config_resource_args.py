from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
def _get_request_config_resource_args(url, content_type=None, decryption_key_hash_sha256=None, encryption_key=None, error_on_missing_key=True, md5_hash=None, size=None, user_request_args=None):
    """Generates metadata for API calls to storage buckets and objects."""
    if not isinstance(url, storage_url.CloudUrl):
        return None
    user_resource_args = getattr(user_request_args, 'resource_args', None)
    new_resource_args = None
    if url.is_bucket():
        if url.scheme in storage_url.VALID_CLOUD_SCHEMES:
            if url.scheme == storage_url.ProviderPrefix.GCS:
                new_resource_args = _GcsBucketConfig()
                if user_resource_args:
                    new_resource_args.autoclass_terminal_storage_class = user_resource_args.autoclass_terminal_storage_class
                    new_resource_args.default_encryption_key = user_resource_args.default_encryption_key
                    new_resource_args.default_event_based_hold = user_resource_args.default_event_based_hold
                    new_resource_args.default_object_acl_file_path = user_resource_args.default_object_acl_file_path
                    new_resource_args.default_object_acl_grants_to_add = user_resource_args.default_object_acl_grants_to_add
                    new_resource_args.default_object_acl_grants_to_remove = user_resource_args.default_object_acl_grants_to_remove
                    new_resource_args.default_storage_class = user_resource_args.default_storage_class
                    new_resource_args.enable_autoclass = user_resource_args.enable_autoclass
                    new_resource_args.enable_per_object_retention = user_resource_args.enable_per_object_retention
                    new_resource_args.enable_hierarchical_namespace = user_resource_args.enable_hierarchical_namespace
                    new_resource_args.placement = user_resource_args.placement
                    new_resource_args.public_access_prevention = user_resource_args.public_access_prevention
                    new_resource_args.recovery_point_objective = user_resource_args.recovery_point_objective
                    new_resource_args.retention_period = user_resource_args.retention_period
                    new_resource_args.retention_period_to_be_locked = user_resource_args.retention_period_to_be_locked
                    new_resource_args.soft_delete_duration = user_resource_args.soft_delete_duration
                    new_resource_args.uniform_bucket_level_access = user_resource_args.uniform_bucket_level_access
            elif url.scheme == storage_url.ProviderPrefix.S3:
                new_resource_args = _S3BucketConfig()
                _check_for_unsupported_s3_fields(user_request_args)
        else:
            new_resource_args = _BucketConfig()
        new_resource_args.location = getattr(user_resource_args, 'location', None)
        new_resource_args.cors_file_path = getattr(user_resource_args, 'cors_file_path', None)
        new_resource_args.labels_file_path = getattr(user_resource_args, 'labels_file_path', None)
        new_resource_args.labels_to_append = getattr(user_resource_args, 'labels_to_append', None)
        new_resource_args.labels_to_remove = getattr(user_resource_args, 'labels_to_remove', None)
        new_resource_args.lifecycle_file_path = getattr(user_resource_args, 'lifecycle_file_path', None)
        new_resource_args.log_bucket = getattr(user_resource_args, 'log_bucket', None)
        new_resource_args.log_object_prefix = getattr(user_resource_args, 'log_object_prefix', None)
        new_resource_args.requester_pays = getattr(user_resource_args, 'requester_pays', None)
        new_resource_args.versioning = getattr(user_resource_args, 'versioning', None)
        new_resource_args.web_error_page = getattr(user_resource_args, 'web_error_page', None)
        new_resource_args.web_main_page_suffix = getattr(user_resource_args, 'web_main_page_suffix', None)
    elif url.is_object():
        if url.scheme == storage_url.ProviderPrefix.GCS:
            new_resource_args = _GcsObjectConfig()
            if user_resource_args:
                new_resource_args.custom_time = user_resource_args.custom_time
                new_resource_args.event_based_hold = user_resource_args.event_based_hold
                new_resource_args.retain_until = user_resource_args.retain_until
                new_resource_args.retention_mode = user_resource_args.retention_mode
                new_resource_args.temporary_hold = user_resource_args.temporary_hold
        elif url.scheme == storage_url.ProviderPrefix.S3:
            new_resource_args = _S3ObjectConfig()
            _check_for_unsupported_s3_fields(user_request_args)
        else:
            new_resource_args = _ObjectConfig()
        new_resource_args.content_type = content_type
        new_resource_args.md5_hash = md5_hash
        new_resource_args.size = size
        new_resource_args.encryption_key = encryption_key or encryption_util.get_encryption_key()
        if decryption_key_hash_sha256:
            new_resource_args.decryption_key = encryption_util.get_decryption_key(decryption_key_hash_sha256, url if error_on_missing_key else None)
        if user_resource_args:
            if user_resource_args.content_type is not None:
                if user_resource_args.content_type:
                    new_resource_args.content_type = user_resource_args.content_type
                else:
                    new_resource_args.content_type = DEFAULT_CONTENT_TYPE
            if user_resource_args.md5_hash is not None:
                new_resource_args.md5_hash = user_resource_args.md5_hash
            new_resource_args.cache_control = user_resource_args.cache_control
            new_resource_args.content_disposition = user_resource_args.content_disposition
            new_resource_args.content_encoding = user_resource_args.content_encoding
            new_resource_args.content_language = user_resource_args.content_language
            new_resource_args.custom_fields_to_set = user_resource_args.custom_fields_to_set
            new_resource_args.custom_fields_to_remove = user_resource_args.custom_fields_to_remove
            new_resource_args.custom_fields_to_update = user_resource_args.custom_fields_to_update
            new_resource_args.preserve_acl = user_resource_args.preserve_acl
            if user_resource_args.storage_class:
                new_resource_args.storage_class = user_resource_args.storage_class.upper()
    if new_resource_args and user_resource_args:
        new_resource_args.acl_file_path = user_resource_args.acl_file_path
        new_resource_args.acl_grants_to_add = user_resource_args.acl_grants_to_add
        new_resource_args.acl_grants_to_remove = user_resource_args.acl_grants_to_remove
    return new_resource_args