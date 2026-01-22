from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_volume
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def app_actions(self, app_current, scope, actions, results):
    app_modify, app_modify_warning = (None, None)
    app_cd_action = self.na_helper.get_cd_action(app_current, self.parameters)
    if app_cd_action == 'create':
        cp_volume_name = self.parameters['name']
        volume, error = rest_volume.get_volume(self.rest_api, self.parameters['vserver'], cp_volume_name)
        self.fail_on_error(error)
        if volume is not None:
            if scope == 'application':
                app_cd_action = 'convert'
                if not self.rest_api.meets_rest_minimum_version(True, 9, 8, 0):
                    msg = 'Error: converting a LUN volume to a SAN application container requires ONTAP 9.8 or better.'
                    self.module.fail_json(msg=msg)
            else:
                msg = "Error: volume '%s' already exists.  Please use a different group name, or use 'application' scope.  scope=%s"
                self.module.fail_json(msg=msg % (cp_volume_name, scope))
    if app_cd_action is not None:
        actions.append('app_%s' % app_cd_action)
    if app_cd_action == 'create':
        self.validate_app_create()
    if app_cd_action is None and app_current is not None:
        app_modify, app_modify_warning = self.app_changes(scope)
        if app_modify:
            actions.append('app_modify')
            results['app_modify'] = dict(app_modify)
    return (app_cd_action, app_modify, app_modify_warning)