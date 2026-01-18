from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
from ansible.module_utils._text import to_native
def zone_to_dict(zone):
    result = dict(id=zone.id, name=zone.name, number_of_record_sets=zone.number_of_record_sets, name_servers=zone.name_servers, tags=zone.tags, type=zone.zone_type.lower(), registration_virtual_networks=[to_native(x.id) for x in zone.registration_virtual_networks] if zone.registration_virtual_networks else None, resolution_virtual_networks=[to_native(x.id) for x in zone.resolution_virtual_networks] if zone.resolution_virtual_networks else None)
    return result