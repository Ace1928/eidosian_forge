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
def update_lbaas_member(self, lbaas_member, lbaas_pool, body=None):
    """Updates a lbaas_member."""
    return self.put(self.lbaas_member_path % (lbaas_pool, lbaas_member), body=body)