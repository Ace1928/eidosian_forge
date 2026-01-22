from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def _check_dirs(module, array):
    if array.get_directories(names=[module.params['source_dir']]).status_code != 200:
        module.fail_json(msg='Source directory {0} does not exist'.format(module.params['source_dir']))
    if array.get_directories(names=[module.params['target_dir']]).status_code != 200:
        module.fail_json(msg='Target directory {0} does not exist'.format(module.params['target_dir']))