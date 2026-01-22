import argparse
import collections
import getpass
import logging
import sys
from urllib import parse as urlparse
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1.identity import v2 as v2_auth
from keystoneauth1.identity import v3 as v3_auth
from keystoneauth1 import loading
from keystoneauth1 import session
from oslo_utils import importutils
import requests
import cinderclient
from cinderclient._i18n import _
from cinderclient import api_versions
from cinderclient import client
from cinderclient import exceptions as exc
from cinderclient import utils
def _discover_service_type(self, discovered_version):
    SERVICE_TYPES = {'1': 'volume', '2': 'volumev2', '3': 'volumev3'}
    major_version = discovered_version.get_major_version()
    service_type = SERVICE_TYPES[major_version]
    return service_type