from __future__ import absolute_import, division, print_function
import time
from ..module_utils.controller_api import ControllerAPIModule
def wait_for_project_update(module, last_request):
    update_project = module.params.get('update_project')
    wait = module.params.get('wait')
    timeout = module.params.get('timeout')
    interval = module.params.get('interval')
    scm_revision_original = last_request['scm_revision']
    if 'current_update' in last_request['summary_fields']:
        running = True
        while running:
            result = module.get_endpoint('/project_updates/{0}/'.format(last_request['summary_fields']['current_update']['id']))['json']
            if module.is_job_done(result['status']):
                time.sleep(1)
                running = False
        if result['status'] != 'successful':
            module.fail_json(msg='Project update failed')
    elif update_project:
        result = module.post_endpoint(last_request['related']['update'])
        if result['status_code'] != 202:
            module.fail_json(msg='Failed to update project, see response for details', response=result)
        if not wait:
            module.exit_json(**module.json_output)
        result_final = module.wait_on_url(url=result['json']['url'], object_name=module.get_item_name(last_request), object_type='Project Update', timeout=timeout, interval=interval)
        module.json_output['changed'] = True
        if result_final['json']['scm_revision'] == scm_revision_original:
            module.json_output['changed'] = False
    module.exit_json(**module.json_output)