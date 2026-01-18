from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def turn_off_promiscuous(self):
    """Disable all promiscuous mode ports, and give them back in a list.
        Returns
        -------
        list
            Contains every port, where promiscuous mode has been turned off
        """
    ports = []
    ports_of_selected_session = []
    for vspan_session in self.dv_switch.config.vspanSession:
        if vspan_session.sourcePortReceived is not None:
            session_ports = vspan_session.sourcePortReceived.portKey
            for port in session_ports:
                if vspan_session.name == self.name:
                    ports_of_selected_session.append(port)
                elif not port in ports:
                    ports.append(port)
        if vspan_session.sourcePortTransmitted is not None:
            session_ports = vspan_session.sourcePortTransmitted.portKey
            for port in session_ports:
                if vspan_session.name == self.name:
                    ports_of_selected_session.append(port)
                elif not port in ports:
                    ports.append(port)
        if vspan_session.destinationPort is not None:
            session_ports = vspan_session.destinationPort.portKey
            for port in session_ports:
                if vspan_session.name == self.name:
                    ports_of_selected_session.append(port)
                elif not port in ports:
                    ports.append(port)
    promiscuous_ports = []
    if ports:
        dv_ports = self.dv_switch.FetchDVPorts(vim.dvs.PortCriteria(portKey=ports))
        for dv_port in dv_ports:
            if dv_port.config.setting.macManagementPolicy.allowPromiscuous:
                self.set_port_security_promiscuous([dv_port.key], False)
                self.modified_ports.update({dv_port.key: True})
                promiscuous_ports.append(dv_port.key)
    if ports_of_selected_session:
        current_dv_ports = self.dv_switch.FetchDVPorts(vim.dvs.PortCriteria(portKey=ports_of_selected_session))
        for dv_port in current_dv_ports:
            if dv_port.config.setting.macManagementPolicy.allowPromiscuous:
                self.set_port_security_promiscuous([dv_port.key], False)
                self.modified_ports.update({dv_port.key: True})
    return promiscuous_ports