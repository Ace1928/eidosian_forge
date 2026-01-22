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
def add_peer_to_bgp_speaker(self, speaker_id, body=None):
    """Adds a peer to BGP speaker."""
    return self.put(self.bgp_speaker_path % speaker_id + '/add_bgp_peer', body=body)