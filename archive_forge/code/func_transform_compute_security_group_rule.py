from osc_lib import exceptions
from openstackclient.i18n import _
def transform_compute_security_group_rule(sg_rule):
    info = {}
    info.update(sg_rule)
    from_port = info.pop('from_port')
    to_port = info.pop('to_port')
    if isinstance(from_port, int) and isinstance(to_port, int):
        port_range = {'port_range': '%u:%u' % (from_port, to_port)}
    elif from_port is None and to_port is None:
        port_range = {'port_range': ''}
    else:
        port_range = {'port_range': '%s:%s' % (from_port, to_port)}
    info.update(port_range)
    if 'cidr' in info['ip_range']:
        info['ip_range'] = info['ip_range']['cidr']
    else:
        info['ip_range'] = ''
    if info['ip_protocol'] is None:
        info['ip_protocol'] = ''
    elif info['ip_protocol'].lower() == 'icmp':
        info['port_range'] = ''
    group = info.pop('group')
    if 'name' in group:
        info['remote_security_group'] = group['name']
    else:
        info['remote_security_group'] = ''
    return info