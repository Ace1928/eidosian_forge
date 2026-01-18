import os
from shlex import quote as pquote
from xml.dom.minidom import parseString
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.py3 import ensure_string
from libcloud.utils.misc import lowercase_keys
from libcloud.common.base import LibcloudConnection, HttpLibResponseProxy

    Debug class to log all HTTP(s) requests as they could be made
    with the curl command.

    :cvar log: file-like object that logs entries are written to.
    