import logging
from keystoneauth1 import adapter
from oslo_utils import importutils
import requests
from urllib import parse as urlparse
from troveclient.apiclient import client
from troveclient import exceptions
from troveclient import service_catalog
def _plugin_auth(self, auth_url):
    return self.auth_plugin.authenticate(self, auth_url)