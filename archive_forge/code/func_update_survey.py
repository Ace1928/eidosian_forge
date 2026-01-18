from __future__ import absolute_import, division, print_function
from ..module_utils.controller_api import ControllerAPIModule
import json
def update_survey(module, last_request):
    spec_endpoint = last_request.get('related', {}).get('survey_spec')
    if module.params.get('survey_spec') == {}:
        response = module.delete_endpoint(spec_endpoint)
        if response['status_code'] != 200:
            module.fail_json(msg='Failed to delete survey: {0}'.format(response['json']))
    else:
        response = module.post_endpoint(spec_endpoint, **{'data': module.params.get('survey_spec')})
        if response['status_code'] != 200:
            module.fail_json(msg='Failed to update survey: {0}'.format(response['json']['error']))