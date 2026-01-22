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
def _get_list_with_added_and_removed_acl_grants(acl_list, resource_args, is_bucket=False, is_default_object_acl=False):
    """Returns shallow copy of ACL policy object with requested changes.

  Args:
    acl_list (list): Contains Apitools ACL objects for buckets or objects.
    resource_args (request_config_factory._ResourceConfig): Contains desired
      changes for the ACL policy.
    is_bucket (bool): Used to determine if ACL for bucket or object. False
      implies a cloud storage object.
    is_default_object_acl (bool): Used to determine if target is default object
      ACL list.

  Returns:
    list: Shallow copy of acl_list with added and removed grants.
  """
    new_acl_list = []
    if is_default_object_acl:
        acl_identifiers_to_remove = set(resource_args.default_object_acl_grants_to_remove or [])
        acl_grants_to_add = resource_args.default_object_acl_grants_to_add or []
    else:
        acl_identifiers_to_remove = set(resource_args.acl_grants_to_remove or [])
        acl_grants_to_add = resource_args.acl_grants_to_add or []
    acl_identifiers_to_add = set((grant['entity'] for grant in acl_grants_to_add))
    found_match = {identifier: False for identifier in acl_identifiers_to_remove}
    for existing_grant in acl_list:
        if properties.VALUES.storage.run_by_gsutil_shim.GetBool():
            matched_identifier = _get_matching_grant_identifier_to_remove_for_shim(existing_grant, acl_identifiers_to_remove)
        elif existing_grant.entity in acl_identifiers_to_remove:
            matched_identifier = existing_grant.entity
        else:
            matched_identifier = None
        if matched_identifier in found_match:
            found_match[matched_identifier] = True
        elif existing_grant.entity not in acl_identifiers_to_add:
            new_acl_list.append(existing_grant)
    unmatched_entities = [k for k, v in found_match.items() if not v]
    if unmatched_entities:
        raise errors.Error('ACL entities marked for removal did not match existing grants: {}'.format(sorted(unmatched_entities)))
    acl_class = metadata_field_converters.get_bucket_or_object_acl_class(is_bucket)
    for new_grant in acl_grants_to_add:
        new_acl_list.append(acl_class(entity=new_grant.get('entity'), role=new_grant.get('role')))
    return new_acl_list