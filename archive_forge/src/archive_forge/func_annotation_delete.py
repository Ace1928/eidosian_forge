from __future__ import absolute_import, division, print_function
from collections import defaultdict
import re
from ansible.module_utils.basic import AnsibleModule
def annotation_delete(module, run_pkgng, package, tag, value):
    _value = annotation_query(module, run_pkgng, package, tag)
    if _value:
        if not module.check_mode:
            rc, out, err = run_pkgng('annotate', '-y', '-D', package, tag)
            if rc != 0:
                module.fail_json(msg='could not delete annotation to %s: %s' % (package, out), stderr=err)
        return True
    return False