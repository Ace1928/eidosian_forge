from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils import six
from ansible.utils.display import Display
from traceback import format_exception
import json
import sys

        :param terms: a list of resources lookups to run.
        :param variables: ansible variables active at the time of the lookup
        :param api_token: API token
        :param project: optional project label
        :param team: optional team label
        :return: a dictionary of resources credentials
        