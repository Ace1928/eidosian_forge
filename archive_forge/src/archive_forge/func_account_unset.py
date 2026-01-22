import io
import logging
import os
import sys
import urllib
from osc_lib import utils
from openstackclient.api import api
def account_unset(self, properties):
    """Unset account properties

        :param dict properties:
            properties to remove from the account
        """
    headers = self._unset_properties(properties, 'X-Remove-Account-Meta-%s')
    if headers:
        self.create('', headers=headers)