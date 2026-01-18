from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleFilterError
from ansible.module_utils.six import string_types
from ansible.module_utils.common._collections_compat import Mapping, Sequence
from ansible.utils.vars import merge_hash
from collections import defaultdict
from operator import itemgetter
 Ansible list filters 