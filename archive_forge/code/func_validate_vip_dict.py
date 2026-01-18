import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def validate_vip_dict(vip_dict, client_manager):
    if 'subnet_id' not in vip_dict:
        raise osc_exc.CommandError('Additional VIPs must include a subnet-id.')
    subnet_id = get_resource_id(client_manager.neutronclient.list_subnets, 'subnets', vip_dict['subnet_id'])
    vip_dict['subnet_id'] = subnet_id
    if 'ip_address' in vip_dict:
        try:
            ipaddress.ip_address(vip_dict['ip_address'])
        except ValueError as e:
            raise osc_exc.CommandError(str(e))