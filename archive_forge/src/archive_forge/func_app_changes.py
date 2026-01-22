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
def app_changes(self, scope):
    app_current, error = self.rest_app.get_application_details('san')
    self.fail_on_error(error)
    app_name = app_current['name']
    provisioned_size = self.na_helper.safe_get(app_current, ['statistics', 'space', 'provisioned'])
    if provisioned_size is None:
        provisioned_size = 0
    if self.debug:
        self.debug['app_current'] = app_current
        self.debug['got'] = copy.deepcopy(app_current)
    app_current = app_current['san']
    app_current.update(app_current['application_components'][0])
    del app_current['application_components']
    comp_name = app_current['name']
    if comp_name != self.parameters['name']:
        msg = 'desired component/volume name: %s does not match existing component name: %s' % (self.parameters['name'], comp_name)
        if scope == 'application':
            self.module.fail_json(msg='Error: ' + msg + '.  scope=%s' % scope)
        return (None, msg + ".  scope=%s, assuming 'lun' scope." % scope)
    app_current['name'] = app_name
    desired = dict(self.parameters['san_application_template'])
    warning = self.fail_on_large_size_reduction(app_current, desired, provisioned_size)
    changed = self.na_helper.changed
    app_modify = self.na_helper.get_modified_attributes(app_current, desired)
    self.validate_app_changes(app_modify, warning)
    if not app_modify:
        self.na_helper.changed = changed
        app_modify = None
    return (app_modify, None)