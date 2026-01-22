from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def _verify_input(self):
    """
        Verifies the parameter input for types and parent correctness and necessary parameters
        """
    try:
        self.entity_class = getattr(VSPK, 'NU{0:s}'.format(self.type))
    except AttributeError:
        self.module.fail_json(msg='Unrecognised type specified')
    if self.module.check_mode:
        return
    if self.parent_type:
        try:
            self.parent_class = getattr(VSPK, 'NU{0:s}'.format(self.parent_type))
        except AttributeError:
            self.module.fail_json(msg='Unrecognised parent type specified')
        fetcher = self.parent_class().fetcher_for_rest_name(self.entity_class.rest_name)
        if fetcher is None:
            self.module.fail_json(msg='Specified parent is not a valid parent for the specified type')
    elif not self.entity_id:
        self.parent_class = VSPK.NUMe
        fetcher = self.parent_class().fetcher_for_rest_name(self.entity_class.rest_name)
        if fetcher is None:
            self.module.fail_json(msg='No parent specified and root object is not a parent for the type')
    if self.command and self.command == 'change_password' and ('password' not in self.properties.keys()):
        self.module.fail_json(msg='command is change_password but the following are missing: password property')