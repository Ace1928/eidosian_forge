from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def _deprecate_setup(self, attr, target, module):
    if target is None:
        target = self
    if not hasattr(target, attr):
        raise ValueError('Target {0} has no attribute {1}'.format(target, attr))
    if module is None:
        if isinstance(target, AnsibleModule):
            module = target
        elif hasattr(target, 'module') and isinstance(target.module, AnsibleModule):
            module = target.module
        else:
            raise ValueError("Failed to automatically discover the AnsibleModule instance. Pass 'module' parameter explicitly.")
    value_attr = '__deprecated_attr_value'
    trigger_attr = '__deprecated_attr_trigger'
    if not hasattr(target, value_attr):
        setattr(target, value_attr, {})
    if not hasattr(target, trigger_attr):
        setattr(target, trigger_attr, {})
    value_dict = getattr(target, value_attr)
    trigger_dict = getattr(target, trigger_attr)
    return (target, module, value_dict, trigger_dict)