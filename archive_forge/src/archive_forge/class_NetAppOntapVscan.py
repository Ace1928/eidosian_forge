from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
class NetAppOntapVscan(object):
    """ enable/disable vscan """

    def __init__(self):
        self.use_rest = False
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(enable=dict(type='bool', default=True), vserver=dict(required=True, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = OntapRestAPI(self.module)
        if self.rest_api.is_rest():
            self.use_rest = True
        elif HAS_NETAPP_LIB is False:
            self.module.fail_json(msg='the python NetApp-Lib module is required')
        else:
            self.server = netapp_utils.setup_na_ontap_zapi(module=self.module, vserver=self.parameters['vserver'])

    def get_vscan(self):
        if self.use_rest:
            params = {'fields': 'svm,enabled', 'svm.name': self.parameters['vserver']}
            api = 'protocols/vscan'
            message, error = self.rest_api.get(api, params)
            if error:
                self.module.fail_json(msg=error)
            return message['records'][0]
        else:
            vscan_status_iter = netapp_utils.zapi.NaElement('vscan-status-get-iter')
            vscan_status_info = netapp_utils.zapi.NaElement('vscan-status-info')
            vscan_status_info.add_new_child('vserver', self.parameters['vserver'])
            query = netapp_utils.zapi.NaElement('query')
            query.add_child_elem(vscan_status_info)
            vscan_status_iter.add_child_elem(query)
            try:
                result = self.server.invoke_successfully(vscan_status_iter, True)
            except netapp_utils.zapi.NaApiError as error:
                self.module.fail_json(msg='Error getting Vscan info for Vserver %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())
            if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
                return result.get_child_by_name('attributes-list').get_child_by_name('vscan-status-info')

    def enable_vscan(self, uuid=None):
        if self.use_rest:
            params = {'svm.name': self.parameters['vserver']}
            data = {'enabled': self.parameters['enable']}
            api = 'protocols/vscan/' + uuid
            dummy, error = self.rest_api.patch(api, data, params)
            if error is not None:
                self.module.fail_json(msg=error)
        else:
            vscan_status_obj = netapp_utils.zapi.NaElement('vscan-status-modify')
            vscan_status_obj.add_new_child('is-vscan-enabled', str(self.parameters['enable']))
            try:
                self.server.invoke_successfully(vscan_status_obj, True)
            except netapp_utils.zapi.NaApiError as error:
                self.module.fail_json(msg='Error Enable/Disabling Vscan: %s' % to_native(error), exception=traceback.format_exc())

    def apply(self):
        changed = False
        current = self.get_vscan()
        if self.use_rest:
            if current['enabled'] != self.parameters['enable']:
                if not self.module.check_mode:
                    self.enable_vscan(current['svm']['uuid'])
                changed = True
        elif current.get_child_content('is-vscan-enabled') != str(self.parameters['enable']).lower():
            if not self.module.check_mode:
                self.enable_vscan()
            changed = True
        self.module.exit_json(changed=changed)