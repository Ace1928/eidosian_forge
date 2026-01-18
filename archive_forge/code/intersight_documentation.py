from __future__ import (absolute_import, division, print_function)
from base64 import b64encode
from email.utils import formatdate
import re
import json
import hashlib
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlencode
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.basic import env_fallback

        GET a resource and return the 1st element found or the full Results list
        