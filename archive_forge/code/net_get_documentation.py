from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
import uuid
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six.moves.urllib.parse import urlsplit
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display

        Determines whether the source and destination file match.

        :return: False if source and dest both exist and have matching sha1 sums, True otherwise.
        