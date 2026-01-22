from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
class NaElementSWConnection(object):

    def __init__(self):
        self.argument_spec = netapp_utils.ontap_sf_host_argument_spec()
        self.argument_spec.update(dict(skip=dict(required=False, type='str', default=None, choices=['mvip', 'svip']), mvip=dict(required=False, type='str', default=None), svip=dict(required=False, type='str', default=None)))
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_if=[('skip', 'svip', ['mvip']), ('skip', 'mvip', ['svip'])], supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.module.params.copy()
        self.msg = ''
        if HAS_SF_SDK is False:
            self.module.fail_json(msg='Unable to import the ElementSW Python SDK')
        else:
            self.elem = netapp_utils.create_sf_connection(self.module, port=442)

    def check_mvip_connection(self):
        """
            Check connection to MVIP

            :return: true if connection was successful, false otherwise.
            :rtype: bool
        """
        try:
            test = self.elem.test_connect_mvip(mvip=self.parameters['mvip'])
            return test.details.connected
        except Exception as e:
            self.msg += 'Error checking connection to MVIP: %s' % to_native(e)
            return False

    def check_svip_connection(self):
        """
            Check connection to SVIP

            :return: true if connection was successful, false otherwise.
            :rtype: bool
        """
        try:
            test = self.elem.test_connect_svip(svip=self.parameters['svip'])
            return test.details.connected
        except Exception as e:
            self.msg += 'Error checking connection to SVIP: %s' % to_native(e)
            return False

    def apply(self):
        passed = False
        if self.parameters.get('skip') is None:
            passed = self.check_mvip_connection()
            passed &= self.check_svip_connection()
        elif self.parameters['skip'] == 'mvip':
            passed |= self.check_svip_connection()
        elif self.parameters['skip'] == 'svip':
            passed |= self.check_mvip_connection()
        if not passed:
            self.module.fail_json(msg=self.msg)
        else:
            self.module.exit_json()