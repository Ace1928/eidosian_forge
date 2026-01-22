from __future__ import absolute_import, division, print_function
import hashlib
import re
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.six.moves import zip
def _diff_strict(self, other):
    updates = list()
    if other and isinstance(other, list) and (len(other) > 0):
        start_other = other[0]
        if start_other.parents:
            for parent in start_other.parents:
                other.insert(0, ConfigLine(parent))
    for index, line in enumerate(self.items):
        try:
            if str(line).strip() != str(other[index]).strip():
                updates.append(line)
        except (AttributeError, IndexError):
            updates.append(line)
    return updates