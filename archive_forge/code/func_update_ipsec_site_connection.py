import inspect
import itertools
import logging
import re
import time
import urllib.parse as urlparse
import debtcollector.renames
from keystoneauth1 import exceptions as ksa_exc
import requests
from neutronclient._i18n import _
from neutronclient import client
from neutronclient.common import exceptions
from neutronclient.common import extension as client_extension
from neutronclient.common import serializer
from neutronclient.common import utils
def update_ipsec_site_connection(self, ipsecsite_conn, body=None):
    """Updates an IPsecSiteConnection."""
    return self.put(self.ipsec_site_connection_path % ipsecsite_conn, body=body)