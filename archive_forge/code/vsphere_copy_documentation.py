from __future__ import absolute_import, division, print_function
import atexit
import errno
import mmap
import os
import socket
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils._text import to_native
from ansible.module_utils.urls import open_url
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
 Constructs a URL path that vSphere accepts reliably 