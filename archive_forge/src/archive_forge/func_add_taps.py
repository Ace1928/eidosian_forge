from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def add_taps(module, brew_path, taps):
    """Adds one or more taps."""
    failed, changed, unchanged, added, msg = (False, False, 0, 0, '')
    for tap in taps:
        failed, changed, msg = add_tap(module, brew_path, tap)
        if failed:
            break
        if changed:
            added += 1
        else:
            unchanged += 1
    if failed:
        msg = 'added: %d, unchanged: %d, error: ' + msg
        msg = msg % (added, unchanged)
    elif added:
        changed = True
        msg = 'added: %d, unchanged: %d' % (added, unchanged)
    else:
        msg = 'added: %d, unchanged: %d' % (added, unchanged)
    return (failed, changed, msg)