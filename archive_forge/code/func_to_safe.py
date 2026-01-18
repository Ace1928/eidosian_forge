from __future__ import (absolute_import, division, print_function)
import itertools
import re
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def to_safe(self, word):
    """Converts 'bad' characters in a string to underscores so they can be used as Ansible groups
        #> ProxmoxInventory.to_safe("foo-bar baz")
        'foo_barbaz'
        """
    regex = '[^A-Za-z0-9\\_]'
    return re.sub(regex, '_', word.replace(' ', ''))