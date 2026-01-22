import io
import logging
import os
import sys
import urllib
from osc_lib import utils
from openstackclient.api import api
def account_show(self):
    """Show account details"""
    response = self._request('HEAD', '')
    data = {}
    properties = self._get_properties(response.headers, 'x-account-meta-')
    if properties:
        data['properties'] = properties
    data['Containers'] = response.headers.get('x-account-container-count')
    data['Objects'] = response.headers.get('x-account-object-count')
    data['Bytes'] = response.headers.get('x-account-bytes-used')
    data['Account'] = self._find_account_id()
    return data