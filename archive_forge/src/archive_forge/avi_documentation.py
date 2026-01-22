from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.plugins.lookup import LookupBase
from ansible.utils.display import Display
from ansible_collections.community.network.plugins.module_utils.network.avi.avi_api import (ApiSession,

    Generic function to handle both /<obj_type>/<obj_uuid> and /<obj_type>
    API resource endpoints.
    