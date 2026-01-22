import copy
import json
import optparse
import os
import pickle
import sys
from urllib import parse
from troveclient.compat import client
from troveclient.compat import exceptions
class AuthedCommandsBase(CommandsBase):
    """Commands that work only with an authenticated client."""

    def __init__(self, parser):
        """Makes sure a token is available somehow and logs in."""
        super(AuthedCommandsBase, self).__init__(parser)
        try:
            self._require('token')
        except ArgumentRequired:
            if self.debug:
                raise
            print('No token argument supplied. Use the "auth login" command to log in and get a token.\n')
            sys.exit(1)
        try:
            self._require('service_url')
        except ArgumentRequired:
            if self.debug:
                raise
            print('No service_url given.\n')
            sys.exit(1)
        self.dbaas = self._get_client()
        self.dbaas.client.auth_token = self.token
        self.dbaas.client.authenticate_with_token(self.token, self.service_url)