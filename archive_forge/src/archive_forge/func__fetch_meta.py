from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import rax_argument_spec, rax_required_together, setup_rax_module
def _fetch_meta(module, container):
    EXIT_DICT['meta'] = dict()
    try:
        for k, v in container.get_metadata().items():
            split_key = k.split(META_PREFIX)[-1]
            EXIT_DICT['meta'][split_key] = v
    except Exception as e:
        module.fail_json(msg=e.message)