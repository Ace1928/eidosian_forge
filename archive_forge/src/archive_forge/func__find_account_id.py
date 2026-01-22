import io
import logging
import os
import sys
import urllib
from osc_lib import utils
from openstackclient.api import api
def _find_account_id(self):
    url_parts = urllib.parse.urlparse(self.endpoint)
    return url_parts.path.split('/')[-1]