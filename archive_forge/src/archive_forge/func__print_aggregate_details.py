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
def _print_aggregate_details(cs, aggregate):
    columns = ['Id', 'Name', 'Availability Zone', 'Hosts', 'Metadata']
    if cs.api_version >= api_versions.APIVersion('2.41'):
        columns.append('UUID')

    def parser_metadata(fields):
        return utils.pretty_choice_dict(getattr(fields, 'metadata', {}) or {})

    def parser_hosts(fields):
        return utils.pretty_choice_list(getattr(fields, 'hosts', []))
    formatters = {'Metadata': parser_metadata, 'Hosts': parser_hosts}
    utils.print_list([aggregate], columns, formatters=formatters)