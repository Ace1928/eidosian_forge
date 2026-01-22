import copy
import logging
from collections import deque, namedtuple
from botocore.compat import accepts_kwargs
from botocore.utils import EVENT_ALIASES
def _replace_subsection(self, sections, old_parts, new_part):
    for i in range(len(sections)):
        if sections[i] == old_parts[0] and sections[i:i + len(old_parts)] == old_parts:
            sections[i:i + len(old_parts)] = [new_part]
            return