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
@staticmethod
def _validate_input_api_version(options):
    if not options.os_volume_api_version:
        api_version = api_versions.APIVersion(api_versions.MAX_VERSION)
    else:
        api_version = api_versions.get_api_version(options.os_volume_api_version)
    return api_version