import argparse
import collections
import datetime
import getpass
import logging
import os
import pprint
import sys
import time
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import timeutils
import novaclient
from novaclient import api_versions
from novaclient import base
from novaclient import client
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import shell
from novaclient import utils
from novaclient.v2 import availability_zones
from novaclient.v2 import quotas
from novaclient.v2 import servers
def _parse_device_spec(device_spec):
    spec_dict = {}
    for arg in device_spec.split(','):
        if '=' in arg:
            spec_dict.update([arg.split('=')])
        else:
            raise argparse.ArgumentTypeError(_("Expected a comma-separated list of key=value pairs. '%s' is not a key=value pair.") % arg)
    return spec_dict