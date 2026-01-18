from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def volume_modify_attributes(self, params):
    """
        modify volume parameter 'export_policy','unix_permissions','snapshot_policy','space_guarantee', 'percent_snapshot_space',
                                'qos_policy_group', 'qos_adaptive_policy_group'
        """
    if self.use_rest:
        return self.volume_modify_attributes_rest(params)
    vol_mod_iter = self.build_zapi_volume_modify_iter(params)
    try:
        result = self.server.invoke_successfully(vol_mod_iter, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        error_msg = to_native(error)
        if 'volume-comp-aggr-attributes' in error_msg:
            error_msg += '. Added info: tiering option requires 9.4 or later.'
        self.wrap_fail_json(msg='Error modifying volume %s: %s' % (self.parameters['name'], error_msg), exception=traceback.format_exc())
    failures = result.get_child_by_name('failure-list')
    if failures is not None:
        error_msgs = [failures.get_child_by_name(return_info).get_child_content('error-message') for return_info in ('volume-modify-iter-info', 'volume-modify-iter-async-info') if failures.get_child_by_name(return_info) is not None]
        if error_msgs and any((x is not None for x in error_msgs)):
            self.wrap_fail_json(msg='Error modifying volume %s: %s' % (self.parameters['name'], ' --- '.join(error_msgs)), exception=traceback.format_exc())
    if self.volume_style == 'flexgroup' or self.parameters['is_infinite']:
        success = self.na_helper.safe_get(result, ['success-list', 'volume-modify-iter-async-info'])
        results = {}
        for key in ('status', 'jobid'):
            if success and success.get_child_by_name(key):
                results[key] = success[key]
        status = results.get('status')
        if status == 'in_progress' and 'jobid' in results:
            if self.parameters['time_out'] == 0:
                return
            error = self.check_job_status(results['jobid'])
            if error is None:
                return
            self.wrap_fail_json(msg='Error when modifying volume: %s' % error)
        self.wrap_fail_json(msg='Unexpected error when modifying volume: result is: %s' % str(result.to_string()))