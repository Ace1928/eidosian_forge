from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import os
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.core.util import debug_output
class _UserBucketArgs(_UserResourceArgs):
    """Contains user flag values affecting cloud bucket settings."""

    def __init__(self, acl_file_path=None, acl_grants_to_add=None, acl_grants_to_remove=None, autoclass_terminal_storage_class=None, cors_file_path=None, default_encryption_key=None, default_event_based_hold=None, default_object_acl_file_path=None, default_object_acl_grants_to_add=None, default_object_acl_grants_to_remove=None, default_storage_class=None, enable_autoclass=None, enable_per_object_retention=None, enable_hierarchical_namespace=None, labels_file_path=None, labels_to_append=None, labels_to_remove=None, lifecycle_file_path=None, location=None, log_bucket=None, log_object_prefix=None, placement=None, public_access_prevention=None, recovery_point_objective=None, requester_pays=None, retention_period=None, retention_period_to_be_locked=False, soft_delete_duration=None, uniform_bucket_level_access=None, versioning=None, web_error_page=None, web_main_page_suffix=None):
        """Initializes class, binding flag values to it."""
        super(_UserBucketArgs, self).__init__(acl_file_path, acl_grants_to_add, acl_grants_to_remove)
        self.autoclass_terminal_storage_class = autoclass_terminal_storage_class
        self.cors_file_path = cors_file_path
        self.default_encryption_key = default_encryption_key
        self.default_event_based_hold = default_event_based_hold
        self.default_object_acl_file_path = default_object_acl_file_path
        self.default_object_acl_grants_to_add = default_object_acl_grants_to_add
        self.default_object_acl_grants_to_remove = default_object_acl_grants_to_remove
        self.default_storage_class = default_storage_class
        self.enable_autoclass = enable_autoclass
        self.enable_per_object_retention = enable_per_object_retention
        self.enable_hierarchical_namespace = enable_hierarchical_namespace
        self.labels_file_path = labels_file_path
        self.labels_to_append = labels_to_append
        self.labels_to_remove = labels_to_remove
        self.lifecycle_file_path = lifecycle_file_path
        self.location = location
        self.log_bucket = log_bucket
        self.log_object_prefix = log_object_prefix
        self.placement = placement
        self.public_access_prevention = public_access_prevention
        self.recovery_point_objective = recovery_point_objective
        self.requester_pays = requester_pays
        self.retention_period = retention_period
        self.retention_period_to_be_locked = retention_period_to_be_locked
        self.soft_delete_duration = soft_delete_duration
        self.uniform_bucket_level_access = uniform_bucket_level_access
        self.versioning = versioning
        self.web_error_page = web_error_page
        self.web_main_page_suffix = web_main_page_suffix

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return super(_UserBucketArgs, self).__eq__(other) and self.autoclass_terminal_storage_class == other.autoclass_terminal_storage_class and (self.cors_file_path == other.cors_file_path) and (self.default_encryption_key == other.default_encryption_key) and (self.default_event_based_hold == other.default_event_based_hold) and (self.default_object_acl_file_path == other.default_object_acl_file_path) and (self.default_object_acl_grants_to_add == other.default_object_acl_grants_to_add) and (self.default_object_acl_grants_to_remove == other.default_object_acl_grants_to_remove) and (self.default_storage_class == other.default_storage_class) and (self.enable_autoclass == other.enable_autoclass) and (self.enable_per_object_retention == other.enable_per_object_retention) and (self.enable_hierarchical_namespace == other.enable_hierarchical_namespace) and (self.labels_file_path == other.labels_file_path) and (self.labels_to_append == other.labels_to_append) and (self.labels_to_remove == other.labels_to_remove) and (self.lifecycle_file_path == other.lifecycle_file_path) and (self.location == other.location) and (self.log_bucket == other.log_bucket) and (self.log_object_prefix == other.log_object_prefix) and (self.placement == other.placement) and (self.public_access_prevention == other.public_access_prevention) and (self.recovery_point_objective == other.recovery_point_objective) and (self.requester_pays == other.requester_pays) and (self.retention_period == other.retention_period) and (self.retention_period_to_be_locked == other.retention_period_to_be_locked) and (self.soft_delete_duration == other.soft_delete_duration) and (self.uniform_bucket_level_access == other.uniform_bucket_level_access) and (self.versioning == other.versioning) and (self.web_error_page == other.web_error_page) and (self.web_main_page_suffix == other.web_main_page_suffix)