from __future__ import absolute_import, division, print_function
import datetime
import re
import time
import tarfile
from ansible.module_utils.urls import fetch_file
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlunparse
@staticmethod
def uri_is_http(uri):
    """Return True if the specified URI is http or https.

        :param str uri: A URI.
        :return: True if the URI is http or https, else False
        :rtype: bool
        """
    parsed_bundle_uri = urlparse(uri)
    return parsed_bundle_uri.scheme.lower() in ['http', 'https']