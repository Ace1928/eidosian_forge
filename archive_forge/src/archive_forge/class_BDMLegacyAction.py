import argparse
import getpass
import io
import json
import logging
import os
from cliff import columns as cliff_columns
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common as network_common
class BDMLegacyAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is None:
            setattr(namespace, self.dest, [])
        dev_name, sep, dev_map = values.partition('=')
        dev_map = dev_map.split(':') if dev_map else dev_map
        if not dev_name or not dev_map or len(dev_map) > 4:
            msg = _("Invalid argument %s; argument must be of form 'dev-name=id[:type[:size[:delete-on-terminate]]]'")
            raise argparse.ArgumentError(self, msg % values)
        mapping = {'device_name': dev_name, 'uuid': dev_map[0], 'source_type': 'volume', 'destination_type': 'volume'}
        if len(dev_map) > 1 and dev_map[1]:
            if dev_map[1] not in ('volume', 'snapshot', 'image'):
                msg = _("Invalid argument %s; 'type' must be one of: volume, snapshot, image")
                raise argparse.ArgumentError(self, msg % values)
            mapping['source_type'] = dev_map[1]
        if len(dev_map) > 2 and dev_map[2]:
            mapping['volume_size'] = dev_map[2]
        if len(dev_map) > 3 and dev_map[3]:
            mapping['delete_on_termination'] = dev_map[3]
        getattr(namespace, self.dest).append(mapping)