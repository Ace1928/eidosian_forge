import io
import logging
import os
import sys
import urllib
from osc_lib import utils
from openstackclient.api import api
def account_set(self, properties):
    """Set account properties

        :param dict properties:
            properties to add or update for the account
        """
    headers = self._set_properties(properties, 'X-Account-Meta-%s')
    if headers:
        self.create('', headers=headers)