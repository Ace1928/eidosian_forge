from __future__ import absolute_import, division, print_function
import atexit
import ansible.module_utils.common._collections_compat as collections_compat
import json
import os
import re
import socket
import ssl
import hashlib
import time
import traceback
import datetime
from collections import OrderedDict
from ansible.module_utils.compat.version import StrictVersion
from random import randint
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.six import integer_types, iteritems, string_types, raise_from
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import unquote
def vcenter_version_at_least(self, version=None):
    """
        Check that the vCenter server is at least a specific version number
        Args:
            version (tuple): a version tuple, for example (6, 7, 0)
        Returns: bool
        """
    if version:
        vc_version = self.content.about.version
        return StrictVersion(vc_version) >= StrictVersion('.'.join(map(str, version)))
    self.module.fail_json(msg='The passed vCenter version: %s is None.' % version)