from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
from apitools.base.py import encoding
from googlecloudsdk.api_lib.edge_cloud.networking import utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
import six
def ModifyToRemoveInterface(self, args, existing):
    """Mutate the router to delete a list of interfaces."""
    input_remove_list = args.interface_names if args.interface_names else []
    input_remove_list = input_remove_list + ([args.interface_name] if args.interface_name else [])
    actual_remove_list = []
    replacement = encoding.CopyProtoMessage(existing)
    existing_router = encoding.CopyProtoMessage(existing)
    for iface in existing_router.interface:
        if iface.name in input_remove_list:
            replacement.interface.remove(iface)
            actual_remove_list.append(iface.name)
    not_found_interface = sorted(set(input_remove_list) - set(actual_remove_list))
    if not_found_interface:
        error_msg = 'interface [{}] not found'.format(', '.join(not_found_interface))
        raise core_exceptions.Error(error_msg)
    return replacement