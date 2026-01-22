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
def _do_hypervisor_list(cs, matching=None, limit=None, marker=None):
    columns = ['ID', 'Hypervisor hostname', 'State', 'Status']
    if matching:
        utils.print_list(cs.hypervisors.search(matching), columns)
    else:
        params = {}
        if limit is not None:
            params['limit'] = limit
        if marker is not None:
            params['marker'] = marker
        utils.print_list(cs.hypervisors.list(False, **params), columns)