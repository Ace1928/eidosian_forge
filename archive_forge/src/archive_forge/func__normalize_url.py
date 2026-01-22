from __future__ import absolute_import, division, print_function
import hashlib
import io
import os
import re
import ssl
import sys
import tarfile
import time
import traceback
import xml.etree.ElementTree as ET
from threading import Thread
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible.module_utils.six.moves.urllib.request import Request, urlopen
from ansible.module_utils.urls import generic_urlparse, open_url, urlparse, urlunparse
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def _normalize_url(self, url):
    """
        The hostname in URLs from vmware may be ``*`` update it accordingly
        """
    url_parts = generic_urlparse(urlparse(url))
    if url_parts.hostname == '*':
        if url_parts.port:
            url_parts.netloc = '%s:%d' % (self.params['hostname'], url_parts.port)
        else:
            url_parts.netloc = self.params['hostname']
    return urlunparse(url_parts.as_list())