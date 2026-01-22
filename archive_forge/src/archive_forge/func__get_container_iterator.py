from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import shim_format_util
import six
def _get_container_iterator(self, container_cloud_url, recursion_level):
    """For recursing into and retrieving the contents of a container.

    Args:
      container_cloud_url (storage_url.CloudUrl): Container URL for recursing
        into.
      recursion_level (int): Determines if iterator should keep recursing.

    Returns:
      BaseFormatWrapper generator.
    """
    new_cloud_url = container_cloud_url.join('*')
    fields_scope = _translate_display_detail_to_fields_scope(self._display_detail, is_bucket_listing=False)
    if self._include_managed_folders:
        managed_folder_setting = folder_util.ManagedFolderSetting.LIST_AS_PREFIXES
    else:
        managed_folder_setting = folder_util.ManagedFolderSetting.DO_NOT_LIST
    iterator = wildcard_iterator.CloudWildcardIterator(new_cloud_url, error_on_missing_key=False, exclude_patterns=self._exclude_patterns, fetch_encrypted_object_hashes=self._fetch_encrypted_object_hashes, fields_scope=fields_scope, halt_on_empty_response=self._halt_on_empty_response, managed_folder_setting=managed_folder_setting, next_page_token=self._next_page_token, object_state=self._object_state)
    return self._recursion_helper(iterator, recursion_level)