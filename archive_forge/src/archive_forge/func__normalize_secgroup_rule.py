from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
def _normalize_secgroup_rule(self, rule):
    ret = utils.Munch()
    rule = rule.copy()
    ret['id'] = rule.pop('id')
    ret['direction'] = rule.pop('direction', 'ingress')
    ret['ethertype'] = rule.pop('ethertype', 'IPv4')
    port_range_min = rule.get('port_range_min', rule.pop('from_port', None))
    if port_range_min == -1:
        port_range_min = None
    if port_range_min is not None:
        port_range_min = int(port_range_min)
    ret['port_range_min'] = port_range_min
    port_range_max = rule.pop('port_range_max', rule.pop('to_port', None))
    if port_range_max == -1:
        port_range_max = None
    if port_range_min is not None:
        port_range_min = int(port_range_min)
    ret['port_range_max'] = port_range_max
    ret['protocol'] = rule.pop('protocol', rule.pop('ip_protocol', None))
    ret['remote_ip_prefix'] = rule.pop('remote_ip_prefix', rule.pop('ip_range', {}).get('cidr', None))
    ret['security_group_id'] = rule.pop('security_group_id', rule.pop('parent_group_id', None))
    ret['remote_group_id'] = rule.pop('remote_group_id', None)
    project_id = rule.pop('tenant_id', '')
    project_id = rule.pop('project_id', project_id)
    ret['location'] = self._get_current_location(project_id=project_id)
    ret['properties'] = rule
    if not self.strict_mode:
        ret['tenant_id'] = project_id
        ret['project_id'] = project_id
        for key, val in ret['properties'].items():
            ret.setdefault(key, val)
    return ret