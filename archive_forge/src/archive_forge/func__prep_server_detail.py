import argparse
import getpass
import io
import json
import logging
import os
from cliff import columns as cliff_columns
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common as network_common
def _prep_server_detail(compute_client, image_client, server, refresh=True):
    """Prepare the detailed server dict for printing

    :param compute_client: a compute client instance
    :param image_client: an image client instance
    :param server: a Server resource
    :param refresh: Flag indicating if ``server`` is already the latest version
                    or if it needs to be refreshed, for example when showing
                    the latest details of a server after creating it.
    :rtype: a dict of server details
    """
    info = server.to_dict()
    if refresh:
        server = utils.find_resource(compute_client.servers, info['id'])
        info.update(server.to_dict())
    column_map = {'access_ipv4': 'accessIPv4', 'access_ipv6': 'accessIPv6', 'admin_password': 'adminPass', 'attached_volumes': 'volumes_attached', 'availability_zone': 'OS-EXT-AZ:availability_zone', 'compute_host': 'OS-EXT-SRV-ATTR:host', 'created_at': 'created', 'disk_config': 'OS-DCF:diskConfig', 'flavor_id': 'flavorRef', 'has_config_drive': 'config_drive', 'host_id': 'hostId', 'fault': 'fault', 'hostname': 'OS-EXT-SRV-ATTR:hostname', 'hypervisor_hostname': 'OS-EXT-SRV-ATTR:hypervisor_hostname', 'instance_name': 'OS-EXT-SRV-ATTR:instance_name', 'is_locked': 'locked', 'kernel_id': 'OS-EXT-SRV-ATTR:kernel_id', 'launch_index': 'OS-EXT-SRV-ATTR:launch_index', 'launched_at': 'OS-SRV-USG:launched_at', 'power_state': 'OS-EXT-STS:power_state', 'project_id': 'tenant_id', 'ramdisk_id': 'OS-EXT-SRV-ATTR:ramdisk_id', 'reservation_id': 'OS-EXT-SRV-ATTR:reservation_id', 'root_device_name': 'OS-EXT-SRV-ATTR:root_device_name', 'task_state': 'OS-EXT-STS:task_state', 'terminated_at': 'OS-SRV-USG:terminated_at', 'updated_at': 'updated', 'user_data': 'OS-EXT-SRV-ATTR:user_data', 'vm_state': 'OS-EXT-STS:vm_state'}
    ignored_columns = {'interface_ip', 'location', 'private_v4', 'private_v6', 'public_v4', 'public_v6', 'block_device_mapping', 'image_id', 'max_count', 'min_count', 'scheduler_hints', 'volumes', 'links'}
    optional_columns = {'admin_password', 'fault', 'flavor_id', 'networks', 'security_groups'}
    data = {}
    for key, value in info.items():
        if key in ignored_columns:
            continue
        if key in optional_columns:
            if info[key] is None:
                continue
        alias = column_map.get(key)
        data[alias or key] = value
    info = data
    image_info = info.get('image', {})
    if image_info and any(image_info.values()):
        image_id = image_info.get('id', '')
        try:
            image = image_client.get_image(image_id)
            info['image'] = '%s (%s)' % (image.name, image_id)
        except Exception:
            info['image'] = image_id
    else:
        info['image'] = IMAGE_STRING_FOR_BFV
    flavor_info = info.get('flavor', {})
    if flavor_info.get('original_name') is None:
        flavor_id = flavor_info.get('id', '')
        try:
            flavor = utils.find_resource(compute_client.flavors, flavor_id)
            info['flavor'] = '%s (%s)' % (flavor.name, flavor_id)
        except Exception:
            info['flavor'] = flavor_id
    else:
        info['flavor'] = format_columns.DictColumn(flavor_info)
    if 'volumes_attached' in info:
        info.update({'volumes_attached': format_columns.ListDictColumn([{k: v for k, v in volume.items() if v is not None and k != 'location'} for volume in info.pop('volumes_attached') or []])})
    if 'security_groups' in info:
        info.update({'security_groups': format_columns.ListDictColumn(info.pop('security_groups'))})
    if 'tags' in info:
        info.update({'tags': format_columns.ListColumn(info.pop('tags'))})
    if 'networks' in info:
        info['addresses'] = format_columns.DictListColumn(info.pop('networks', {}))
    else:
        info['addresses'] = AddressesColumn(info.get('addresses', {}))
    info['properties'] = format_columns.DictColumn(info.pop('metadata'))
    if 'tenant_id' in info:
        info['project_id'] = info.pop('tenant_id')
    if 'OS-EXT-STS:power_state' in info:
        info['OS-EXT-STS:power_state'] = PowerStateColumn(info['OS-EXT-STS:power_state'])
    return info