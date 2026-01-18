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
def update_fwaas_firewall_rule(self, firewall_rule, body=None):
    """Updates a firewall rule"""
    return self.put(self.fwaas_firewall_rule_path % firewall_rule, body=body)