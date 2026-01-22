from __future__ import absolute_import, division, print_function
from collections import defaultdict
import re
from ansible.module_utils.basic import AnsibleModule
def annotation_add(module, run_pkgng, package, tag, value):
    _value = annotation_query(module, run_pkgng, package, tag)
    if not _value:
        if not module.check_mode:
            rc, out, err = run_pkgng('annotate', '-y', '-A', package, tag, data=value, binary_data=True)
            if rc != 0:
                module.fail_json(msg='could not annotate %s: %s' % (package, out), stderr=err)
        return True
    elif _value != value:
        module.fail_json(msg='failed to annotate %s, because %s is already set to %s, but should be set to %s' % (package, tag, _value, value))
        return False
    else:
        return False