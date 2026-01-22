import io
import logging
import os
import sys
import urllib
from osc_lib import utils
from openstackclient.api import api
def _unset_properties(self, properties, header_tag):
    headers = {}
    for k in properties:
        header_name = header_tag % k
        headers[header_name] = 'x'
    return headers