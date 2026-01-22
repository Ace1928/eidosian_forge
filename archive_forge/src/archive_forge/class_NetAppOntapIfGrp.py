from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapIfGrp:
    """
        Create, Modifies and Destroys a IfGrp
    """

    def __init__(self):
        """
            Initialize the Ontap IfGrp class
        """
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), distribution_function=dict(required=False, type='str', choices=['mac', 'ip', 'sequential', 'port']), name=dict(required=False, type='str'), mode=dict(required=False, type='str'), node=dict(required=True, type='str'), ports=dict(required=False, type='list', elements='str', aliases=['port']), from_lag_ports=dict(required=False, type='list', elements='str'), broadcast_domain=dict(required=False, type='str'), ipspace=dict(required=False, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_if=[('state', 'present', ['distribution_function', 'mode'])], required_together=[['broadcast_domain', 'ipspace']], supports_check_mode=True)
        self.current_records = []
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = OntapRestAPI(self.module)
        unsupported_rest_properties = ['name']
        self.use_rest = self.rest_api.is_rest_supported_properties(self.parameters, unsupported_rest_properties)
        if self.use_rest:
            if 'ports' not in self.parameters:
                error_msg = 'Error: ports is a required field with REST'
                self.module.fail_json(msg=error_msg)
            required_options = ['broadcast_domain', 'ipspace']
            min_ontap_98 = self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 8, 0)
            if not min_ontap_98 and (not any((x in self.parameters for x in required_options))):
                error_msg = "'%s' are mandatory fields with ONTAP 9.6 and 9.7" % ', '.join(required_options)
                self.module.fail_json(msg=error_msg)
        else:
            if not netapp_utils.has_netapp_lib():
                self.module.fail_json(msg=netapp_utils.netapp_lib_is_required())
            if 'name' not in self.parameters:
                self.module.fail_json('Error: name is a required field with ZAPI.')
            if 'broadcast_domain' in self.parameters or 'ipspace' in self.parameters or 'from_lag_ports' in self.parameters:
                msg = 'Using ZAPI and ignoring options - broadcast_domain, ipspace and from_lag_ports'
                self.module.warn(msg)
                self.parameters.pop('broadcast_domain', None)
                self.parameters.pop('ipspace', None)
                self.parameters.pop('from_lag_ports', None)
            self.server = netapp_utils.setup_na_ontap_zapi(module=self.module)

    def get_if_grp(self):
        """
        Return details about the if_group
        :param:
            name : Name of the if_group

        :return: Details about the if_group. None if not found.
        :rtype: dict
        """
        if_group_iter = netapp_utils.zapi.NaElement('net-port-get-iter')
        if_group_info = netapp_utils.zapi.NaElement('net-port-info')
        if_group_info.add_new_child('port', self.parameters['name'])
        if_group_info.add_new_child('port-type', 'if_group')
        if_group_info.add_new_child('node', self.parameters['node'])
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(if_group_info)
        if_group_iter.add_child_elem(query)
        try:
            result = self.server.invoke_successfully(if_group_iter, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error getting if_group %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        return_value = None
        if result.get_child_by_name('num-records') and int(result['num-records']) >= 1:
            if_group_attributes = result['attributes-list']['net-port-info']
            return_value = {'name': if_group_attributes['port'], 'distribution_function': if_group_attributes['ifgrp-distribution-function'], 'mode': if_group_attributes['ifgrp-mode'], 'node': if_group_attributes['node']}
        return return_value

    def get_if_grp_rest(self, ports, allow_partial_match, force=False):
        api = 'network/ethernet/ports'
        query = {'type': 'lag', 'node.name': self.parameters['node']}
        fields = 'name,node,uuid,broadcast_domain,lag'
        error = None
        if not self.current_records or force:
            self.current_records, error = rest_generic.get_0_or_more_records(self.rest_api, api, query, fields)
        if error:
            self.module.fail_json(msg=error)
        if self.current_records:
            current_ifgrp = self.get_if_grp_current(self.current_records, ports)
            if current_ifgrp:
                exact_match = self.check_exact_match(ports, current_ifgrp['ports'])
                if exact_match or allow_partial_match:
                    return (current_ifgrp, exact_match)
        return (None, None)

    def check_exact_match(self, desired_ports, current_ifgrp):
        matched = set(desired_ports) == set(current_ifgrp)
        if not matched:
            self.rest_api.log_debug(0, 'found LAG with partial match of ports: %s but current is %s' % (desired_ports, current_ifgrp))
        return matched

    def get_if_grp_current(self, records, ports):
        desired_ifgrp_in_current = []
        for record in records:
            if 'member_ports' in record['lag']:
                current_port_list = [port['name'] for port in record['lag']['member_ports']]
                for current_port in current_port_list:
                    if current_port in ports:
                        desired_ifgrp_in_current.append(self.get_if_grp_detail(record, current_port_list))
                        break
        if len(desired_ifgrp_in_current) > 1 and self.parameters['state'] == 'present':
            error_msg = "'%s' are in different LAGs" % ', '.join(ports)
            self.module.fail_json(msg=error_msg)
        elif len(desired_ifgrp_in_current) == 1:
            return desired_ifgrp_in_current[0]
        return None

    def get_if_grp_detail(self, record, current_port_list):
        current = {'node': record['node']['name'], 'uuid': record['uuid'], 'name': record['name'], 'ports': current_port_list}
        if record.get('broadcast_domain'):
            current['broadcast_domain'] = record['broadcast_domain']['name']
            current['ipspace'] = record['broadcast_domain']['ipspace']['name']
        return current

    def get_if_grp_ports(self):
        """
        Return ports of the if_group
        :param:
            name : Name of the if_group
        :return: Ports of the if_group. None if not found.
        :rtype: dict
        """
        if_group_iter = netapp_utils.zapi.NaElement('net-port-ifgrp-get')
        if_group_iter.add_new_child('ifgrp-name', self.parameters['name'])
        if_group_iter.add_new_child('node', self.parameters['node'])
        try:
            result = self.server.invoke_successfully(if_group_iter, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error getting if_group ports %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        port_list = []
        if result.get_child_by_name('attributes'):
            if_group_attributes = result['attributes']['net-ifgrp-info']
            if if_group_attributes.get_child_by_name('ports'):
                ports = if_group_attributes.get_child_by_name('ports').get_children()
                for each in ports:
                    port_list.append(each.get_content())
        return {'ports': port_list}

    def create_if_grp(self):
        """
        Creates a new ifgrp
        """
        if self.use_rest:
            return self.create_if_grp_rest()
        route_obj = netapp_utils.zapi.NaElement('net-port-ifgrp-create')
        route_obj.add_new_child('distribution-function', self.parameters['distribution_function'])
        route_obj.add_new_child('ifgrp-name', self.parameters['name'])
        route_obj.add_new_child('mode', self.parameters['mode'])
        route_obj.add_new_child('node', self.parameters['node'])
        try:
            self.server.invoke_successfully(route_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error creating if_group %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        if self.parameters.get('ports') is not None:
            for port in self.parameters.get('ports'):
                self.add_port_to_if_grp(port)

    def create_if_grp_rest(self):
        api = 'network/ethernet/ports'
        body = {'type': 'lag', 'node': {'name': self.parameters['node']}, 'lag': {'mode': self.parameters['mode'], 'distribution_policy': self.parameters['distribution_function']}}
        if self.parameters.get('ports') is not None:
            body['lag']['member_ports'] = self.build_member_ports()
        if 'broadcast_domain' in self.parameters:
            body['broadcast_domain'] = {'name': self.parameters['broadcast_domain']}
            body['broadcast_domain']['ipspace'] = {'name': self.parameters['ipspace']}
        dummy, error = rest_generic.post_async(self.rest_api, api, body)
        if error:
            self.module.fail_json(msg=error)

    def delete_if_grp(self, uuid=None):
        """
        Deletes a ifgrp
        """
        if self.use_rest:
            api = 'network/ethernet/ports'
            dummy, error = rest_generic.delete_async(self.rest_api, api, uuid)
            if error:
                self.module.fail_json(msg=error)
        else:
            route_obj = netapp_utils.zapi.NaElement('net-port-ifgrp-destroy')
            route_obj.add_new_child('ifgrp-name', self.parameters['name'])
            route_obj.add_new_child('node', self.parameters['node'])
            try:
                self.server.invoke_successfully(route_obj, True)
            except netapp_utils.zapi.NaApiError as error:
                self.module.fail_json(msg='Error deleting if_group %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def add_port_to_if_grp(self, port):
        """
        adds port to a ifgrp
        """
        route_obj = netapp_utils.zapi.NaElement('net-port-ifgrp-add-port')
        route_obj.add_new_child('ifgrp-name', self.parameters['name'])
        route_obj.add_new_child('port', port)
        route_obj.add_new_child('node', self.parameters['node'])
        try:
            self.server.invoke_successfully(route_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error adding port %s to if_group %s: %s' % (port, self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def modify_ports(self, current_ports):
        add_ports = set(self.parameters['ports']) - set(current_ports)
        remove_ports = set(current_ports) - set(self.parameters['ports'])
        for port in add_ports:
            self.add_port_to_if_grp(port)
        for port in remove_ports:
            self.remove_port_to_if_grp(port)

    def modify_ports_rest(self, modify, uuid):
        api = 'network/ethernet/ports'
        body = {}
        if 'ports' in modify:
            member_ports = self.build_member_ports()
            body['lag'] = {'member_ports': member_ports}
        if 'broadcast_domain' in modify or 'ipspace' in modify:
            broadcast_domain = modify['broadcast_domain'] if 'broadcast_domain' in modify else self.parameters['broadcast_domain']
            ipspace = modify['ipspace'] if 'ipspace' in modify else self.parameters['ipspace']
            body['broadcast_domain'] = {'name': broadcast_domain}
            body['broadcast_domain']['ipspace'] = {'name': ipspace}
        dummy, error = rest_generic.patch_async(self.rest_api, api, uuid, body)
        if error:
            self.module.fail_json(msg=error)

    def build_member_ports(self):
        member_ports = []
        for port in self.parameters['ports']:
            port_detail = {'name': port, 'node': {'name': self.parameters['node']}}
            member_ports.append(port_detail)
        return member_ports

    def remove_port_to_if_grp(self, port):
        """
        removes port from a ifgrp
        """
        route_obj = netapp_utils.zapi.NaElement('net-port-ifgrp-remove-port')
        route_obj.add_new_child('ifgrp-name', self.parameters['name'])
        route_obj.add_new_child('port', port)
        route_obj.add_new_child('node', self.parameters['node'])
        try:
            self.server.invoke_successfully(route_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error removing port %s to if_group %s: %s' % (port, self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def apply(self):
        current, exact_match, modify, rename = (None, True, None, None)
        response = None
        if not self.use_rest:
            current = self.get_if_grp()
        elif self.use_rest:
            current, exact_match = self.get_if_grp_rest(self.parameters.get('ports'), allow_partial_match=True)
        cd_action = self.na_helper.get_cd_action(current if exact_match else None, self.parameters)
        if cd_action == 'create' and self.use_rest:
            if self.parameters.get('from_lag_ports') is not None:
                from_ifgrp, dummy = self.get_if_grp_rest(self.parameters['from_lag_ports'], allow_partial_match=False)
                if not from_ifgrp:
                    error_msg = "Error: cannot find LAG matching from_lag_ports: '%s'." % self.parameters['from_lag_ports']
                    self.module.fail_json(msg=error_msg)
                rename = True
                current = from_ifgrp
            elif not exact_match and current:
                rename = True
            if rename:
                cd_action = None
        if cd_action is None and self.parameters['state'] == 'present':
            current_ports = self.get_if_grp_ports() if not self.use_rest else current
            modify = self.na_helper.get_modified_attributes(current_ports, self.parameters)
        if self.na_helper.changed and (not self.module.check_mode):
            uuid = current['uuid'] if current and self.use_rest else None
            if cd_action == 'create':
                self.create_if_grp()
                if self.use_rest:
                    response, exact_match = self.get_if_grp_rest(self.parameters.get('ports'), allow_partial_match=True, force=True)
            elif cd_action == 'delete':
                self.delete_if_grp(uuid)
            elif modify:
                if self.use_rest:
                    self.modify_ports_rest(modify, uuid)
                else:
                    self.modify_ports(current_ports['ports'])
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify, response=response)
        self.module.exit_json(**result)