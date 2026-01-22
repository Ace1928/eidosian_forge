from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.common.dict_transformations import recursive_diff
def _get_baseline_payload(module, rest_obj):
    cat_name = module.params.get('catalog_name')
    cat_id, repo_id = get_catrepo_ids(module, cat_name, rest_obj)
    if cat_id is None or repo_id is None:
        module.fail_json(msg='No Catalog with name {0} found'.format(cat_name))
    targets = get_target_list(module, rest_obj)
    if targets is None:
        module.fail_json(msg=NO_TARGETS_MESSAGE)
    baseline_name = module.params.get('baseline_name')
    baseline_payload = {'Name': baseline_name, 'CatalogId': cat_id, 'RepositoryId': repo_id, 'Targets': targets}
    baseline_payload['Description'] = module.params.get('baseline_description')
    baseline_payload['FilterNoRebootRequired'] = module.params.get('filter_no_reboot_required')
    de = module.params.get('downgrade_enabled')
    baseline_payload['DowngradeEnabled'] = de if de is not None else True
    sfb = module.params.get('is_64_bit')
    baseline_payload['Is64Bit'] = sfb if sfb is not None else True
    return baseline_payload