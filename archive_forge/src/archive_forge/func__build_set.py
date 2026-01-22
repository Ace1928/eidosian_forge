from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils import arguments, errors, utils
def _build_set(builds):
    return set(((b.get('sha512'), b.get('url'), frozenset((b.get('headers', {}) or {}).items()), frozenset(b.get('filters', []) or [])) for b in builds))