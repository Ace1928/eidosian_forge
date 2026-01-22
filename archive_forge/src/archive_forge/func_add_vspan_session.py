from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def add_vspan_session(self):
    """Calls the necessary functions to create a VSpanSession"""
    results = dict(changed=False, result='')
    promiscous_ports = self.turn_off_promiscuous()
    if not self.check_if_session_name_is_free():
        self.module.fail_json(msg='There is another VSpan Session with the name: {0:s}.'.format(self.name))
    dv_ports = None
    ports = [str(self.source_port_received), str(self.source_port_transmitted), str(self.destination_port)]
    if ports:
        dv_ports = self.dv_switch.FetchDVPorts(vim.dvs.PortCriteria(portKey=ports))
    for dv_port in dv_ports:
        if dv_port.config.setting.macManagementPolicy.allowPromiscuous:
            self.set_port_security_promiscuous([dv_port.key], False)
            self.modified_ports.update({dv_port.key: True})
    self.create_vspan_session()
    if self.session_type == 'dvPortMirror' or self.session_type == 'remoteMirrorDest':
        self.set_port_security_promiscuous([str(self.destination_port)], True)
    if promiscous_ports:
        self.set_port_security_promiscuous(promiscous_ports, True)
    results['changed'] = True
    results['result'] = 'Mirroring session has been created.'
    return results