from __future__ import absolute_import, division, print_function
import ast
import json
import operator
import re
import socket
from copy import deepcopy
from functools import reduce  # forward compatibility for Python 3
from itertools import chain
from ansible.module_utils import basic
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems, string_types
class EntityCollection(Entity):
    """Extends ```Entity``` to handle a list of dicts"""

    def __call__(self, iterable, strict=True):
        if iterable is None:
            iterable = [super(EntityCollection, self).__call__(self._module.params, strict)]
        if not isinstance(iterable, (list, tuple)):
            self._module.fail_json(msg='value must be an iterable')
        return [super(EntityCollection, self).__call__(i, strict) for i in iterable]