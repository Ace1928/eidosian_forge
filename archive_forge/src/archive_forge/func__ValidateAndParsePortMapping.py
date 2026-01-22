from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import re
import enum
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _ValidateAndParsePortMapping(port_mappings):
    """Parses and validates port mapping."""
    ports_config = []
    for port_mapping in port_mappings:
        mapping_match = re.match('^(\\d+):(\\d+):(\\S+)$', port_mapping)
        if not mapping_match:
            raise calliope_exceptions.InvalidArgumentException('--port-mappings', 'Port mappings should follow PORT:TARGET_PORT:PROTOCOL format.')
        port, target_port, protocol = mapping_match.groups()
        if protocol not in ALLOWED_PROTOCOLS:
            raise calliope_exceptions.InvalidArgumentException('--port-mappings', 'Protocol should be one of [{0}]'.format(', '.join(ALLOWED_PROTOCOLS)))
        ports_config.append({'containerPort': int(target_port), 'hostPort': int(port), 'protocol': protocol})
    return ports_config