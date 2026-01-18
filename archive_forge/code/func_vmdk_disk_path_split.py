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
def vmdk_disk_path_split(self, vmdk_path):
    """
        Takes a string in the format

            [datastore_name] path/to/vm_name.vmdk

        Returns a tuple with multiple strings:

        1. datastore_name: The name of the datastore (without brackets)
        2. vmdk_fullpath: The "path/to/vm_name.vmdk" portion
        3. vmdk_filename: The "vm_name.vmdk" portion of the string (os.path.basename equivalent)
        4. vmdk_folder: The "path/to/" portion of the string (os.path.dirname equivalent)
        """
    try:
        datastore_name = re.match('^\\[(.*?)\\]', vmdk_path, re.DOTALL).groups()[0]
        vmdk_fullpath = re.match('\\[.*?\\] (.*)$', vmdk_path).groups()[0]
        vmdk_filename = os.path.basename(vmdk_fullpath)
        vmdk_folder = os.path.dirname(vmdk_fullpath)
        return (datastore_name, vmdk_fullpath, vmdk_filename, vmdk_folder)
    except (IndexError, AttributeError) as e:
        self.module.fail_json(msg="Bad path '%s' for filename disk vmdk image: %s" % (vmdk_path, to_native(e)))