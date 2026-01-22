from __future__ import absolute_import, division, print_function
import os
import pwd
import os.path
import tempfile
import re
import shlex
from operator import itemgetter
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
class keydict(dict):
    """ a dictionary that maintains the order of keys as they are added

    This has become an abuse of the dict interface.  Probably should be
    rewritten to be an entirely custom object with methods instead of
    bracket-notation.

    Our requirements are for a data structure that:
    * Preserves insertion order
    * Can store multiple values for a single key.

    The present implementation has the following functions used by the rest of
    the code:

    * __setitem__(): to add a key=value.  The value can never be disassociated
      with the key, only new values can be added in addition.
    * items(): to retrieve the key, value pairs.

    Other dict methods should work but may be surprising.  For instance, there
    will be multiple keys that are the same in keys() and __getitem__() will
    return a list of the values that have been set via __setitem__.
    """

    def __init__(self, *args, **kw):
        super(keydict, self).__init__(*args, **kw)
        self.itemlist = list(super(keydict, self).keys())

    def __setitem__(self, key, value):
        self.itemlist.append(key)
        if key in self:
            self[key].append(value)
        else:
            super(keydict, self).__setitem__(key, [value])

    def __iter__(self):
        return iter(self.itemlist)

    def keys(self):
        return self.itemlist

    def _item_generator(self):
        indexes = {}
        for key in self.itemlist:
            if key in indexes:
                indexes[key] += 1
            else:
                indexes[key] = 0
            yield (key, self[key][indexes[key]])

    def iteritems(self):
        raise NotImplementedError("Do not use this as it's not available on py3")

    def items(self):
        return list(self._item_generator())

    def itervalues(self):
        raise NotImplementedError("Do not use this as it's not available on py3")

    def values(self):
        return [item[1] for item in self.items()]