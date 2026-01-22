from __future__ import (absolute_import, division, print_function)
import abc
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import PY3
from ansible.module_utils.urls import fetch_url, open_url, NoSSLError, ConnectionError
import ansible.module_utils.six.moves.urllib.error as urllib_error
@six.add_metaclass(abc.ABCMeta)
class HTTPHelper(object):

    @abc.abstractmethod
    def fetch_url(self, url, method='GET', headers=None, data=None, timeout=None):
        """
        Execute a HTTP request and return a tuple (response_content, info).

        In case of errors, either raise NetworkError or terminate the program (for modules only!).
        """