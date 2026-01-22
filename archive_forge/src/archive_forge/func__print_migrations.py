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
def _print_migrations(cs, migrations):
    fields = ['Source Node', 'Dest Node', 'Source Compute', 'Dest Compute', 'Dest Host', 'Status', 'Instance UUID', 'Old Flavor', 'New Flavor', 'Created At', 'Updated At']

    def old_flavor(migration):
        return migration.old_instance_type_id

    def new_flavor(migration):
        return migration.new_instance_type_id

    def migration_type(migration):
        return migration.migration_type
    formatters = {'Old Flavor': old_flavor, 'New Flavor': new_flavor}
    if cs.api_version >= api_versions.APIVersion('2.59'):
        fields.insert(0, 'UUID')
    if cs.api_version >= api_versions.APIVersion('2.23'):
        fields.insert(0, 'Id')
        fields.append('Type')
        formatters.update({'Type': migration_type})
    if cs.api_version >= api_versions.APIVersion('2.80'):
        fields.append('Project ID')
        fields.append('User ID')
    utils.print_list(migrations, fields, formatters)