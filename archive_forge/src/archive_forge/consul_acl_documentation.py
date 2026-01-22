from __future__ import absolute_import, division, print_function
from collections import defaultdict
from ansible.module_utils.basic import to_text, AnsibleModule

        Adds the given rule to this collection.
        :param rule: model of a rule
        :raises ValueError: raised if there already exists a rule for a given scope and pattern
        