from __future__ import (absolute_import, division, print_function)
import csv
import os
import json
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_all_data_with_pagination, strip_substr_dict
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.common.dict_transformations import recursive_diff
from datetime import datetime
def transform_existing_policy_data(policy):
    pdata = policy.get('PolicyData')
    undiscovered = pdata.get('UndiscoveredTargets')
    if undiscovered:
        pdata['UndiscoveredTargets'] = [x.get('TargetAddress') for x in undiscovered]
    actions = pdata.get('Actions')
    if actions:
        for action in actions:
            if action.get('Name') == 'RemoteCommand':
                action['ParameterDetails'] = dict(((str(act_param.get('Name')).rstrip('1'), act_param.get('Value')) for act_param in action.get('ParameterDetails', [])))
            else:
                action['ParameterDetails'] = dict(((act_param.get('Name'), act_param.get('Value')) for act_param in action.get('ParameterDetails', [])))
            action.pop('Id', None)
        pdata['Actions'] = dict(((x.get('Name'), x) for x in actions))
    catalogs = pdata.get('Catalogs')
    pdata['Catalogs'] = dict(((x.get('CatalogName'), x) for x in catalogs))
    for pol_data in pdata.values():
        if isinstance(pol_data, list):
            pol_data.sort()
    messages = pdata.get('MessageIds', [])
    pdata['MessageIds'] = [m.strip("'") for m in messages]