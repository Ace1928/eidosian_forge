from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.load_balancers import BoundLoadBalancer
@staticmethod
def _prepare_target_result(target):
    result = {'type': to_native(target.type), 'use_private_ip': target.use_private_ip}
    if target.type == 'server':
        result['server'] = to_native(target.server.name)
    elif target.type == 'label_selector':
        result['label_selector'] = to_native(target.label_selector.selector)
    elif target.type == 'ip':
        result['ip'] = to_native(target.ip.ip)
    if target.health_status is not None:
        result['health_status'] = [{'listen_port': item.listen_port, 'status': item.status} for item in target.health_status]
    return result