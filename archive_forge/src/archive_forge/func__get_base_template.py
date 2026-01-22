from __future__ import (absolute_import, division, print_function)
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_base_template(self):
    templates = self._connection.system_service().templates_service().list()
    if not templates:
        return None
    template_name = self.param('name')
    named_templates = [t for t in templates if t.name == template_name]
    if not named_templates:
        return None
    base_template = min(named_templates, key=lambda x: x.version.version_number)
    return otypes.Template(id=base_template.id)