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
def _merge_usage(usage, next_usage):
    usage.server_usages.extend(next_usage.server_usages)
    usage.total_hours += next_usage.total_hours
    usage.total_memory_mb_usage += next_usage.total_memory_mb_usage
    usage.total_vcpus_usage += next_usage.total_vcpus_usage
    usage.total_local_gb_usage += next_usage.total_local_gb_usage