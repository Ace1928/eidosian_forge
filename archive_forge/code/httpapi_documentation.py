from __future__ import absolute_import, division, print_function
from io import BytesIO
from ansible.errors import AnsibleConnectionFailure
from ansible.module_utils._text import to_bytes
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import cPickle
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.urls import open_url
from ansible.playbook.play_context import PlayContext
from ansible.plugins.connection import ensure_connect
from ansible.plugins.loader import httpapi_loader
from ansible.release import __version__ as ANSIBLE_CORE_VERSION
from ansible_collections.ansible.netcommon.plugins.plugin_utils.connection_base import (
from ansible_collections.ansible.netcommon.plugins.plugin_utils.version import Version
This method enables wait_for_connection to work.

        The sole purpose of this method is to raise an exception if the API's URL
        cannot be reached. As such, it does not do anything except attempt to
        request the root URL with no error handling.
        