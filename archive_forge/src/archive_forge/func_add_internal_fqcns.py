from __future__ import (absolute_import, division, print_function)
def add_internal_fqcns(names):
    """
    Given a sequence of action/module names, returns a list of these names
    with the same names with the prefixes `ansible.builtin.` and
    `ansible.legacy.` added for all names that are not already FQCNs.
    """
    result = []
    for name in names:
        result.append(name)
        if '.' not in name:
            result.append('ansible.builtin.%s' % name)
            result.append('ansible.legacy.%s' % name)
    return result