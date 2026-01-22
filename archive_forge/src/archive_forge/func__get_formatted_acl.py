from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from apitools.base.py import encoding
from googlecloudsdk.command_lib.storage.resources import full_resource_formatter
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
def _get_formatted_acl(acl):
    """Removes unnecessary fields from acl."""
    if acl is None:
        return acl
    formatted_acl = []
    for acl_entry in acl:
        acl_entry_copy = acl_entry.copy()
        if acl_entry_copy.get('kind') == 'storage#objectAccessControl':
            acl_entry_copy.pop('object', None)
            acl_entry_copy.pop('generation', None)
        acl_entry_copy.pop('kind', None)
        acl_entry_copy.pop('bucket', None)
        acl_entry_copy.pop('id', None)
        acl_entry_copy.pop('selfLink', None)
        acl_entry_copy.pop('etag', None)
        formatted_acl.append(acl_entry_copy)
    return formatted_acl