from __future__ import absolute_import, division, print_function
import os
import re
from ansible.errors import AnsibleFilterError
from ansible.module_utils._text import to_native
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.six import iteritems, string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import Template

The parse_xml plugin code
