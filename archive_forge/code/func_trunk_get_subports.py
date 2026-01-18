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
def trunk_get_subports(self, trunk, **_params):
    """Fetch a list of all subports attached to given trunk."""
    return self.get(self.subports_path % trunk, params=_params)