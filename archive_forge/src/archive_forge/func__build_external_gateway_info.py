from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def _build_external_gateway_info(self, ext_gateway_net_id, enable_snat, ext_fixed_ips):
    info = {}
    if ext_gateway_net_id:
        info['network_id'] = ext_gateway_net_id
    if enable_snat is not None:
        info['enable_snat'] = enable_snat
    if ext_fixed_ips:
        info['external_fixed_ips'] = ext_fixed_ips
    if info:
        return info
    return None